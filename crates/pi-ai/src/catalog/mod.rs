//! Checked-in built-in model catalog and tolerant override merge.
//!
//! Runtime loads [`data/builtin-models.json`](../../data/builtin-models.json)
//! through `include_str!` only. No Bun, Node, or other process is required to
//! resolve the catalog.

mod merge;

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::Model;

pub use merge::{apply_model_override, effective_models, json_merge};

/// Logical path of the compiled-in catalog payload.
pub const BUILTIN_MODELS_PATH: &str = "data/builtin-models.json";

const BUILTIN_MODELS_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/builtin-models.json"
));

/// Provider → model-id → model metadata for the built-in catalog.
pub type BuiltinModels = BTreeMap<String, BTreeMap<String, Model>>;

/// Persisted dynamic catalog overlay for one provider.
///
/// An explicit entry (including `models: []`) replaces built-ins for that
/// provider. A missing entry leaves built-ins in place.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsStoreEntry {
    /// Models returned for the provider when this entry is present.
    pub models: Vec<Model>,
    /// Unix epoch milliseconds of the last completed remote check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_at: Option<i64>,
}

/// Path-rich catalog load/merge failure.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum CatalogError {
    /// JSON parse failure at a catalog path.
    #[error("{path}: {message}")]
    Parse {
        /// JSON path or file path where parsing failed.
        path: String,
        /// Underlying parse message.
        message: String,
    },
    /// Required model fields failed validation at a catalog path.
    #[error("{path}: {message}")]
    Validation {
        /// JSON path of the invalid model or field.
        path: String,
        /// Validation message.
        message: String,
    },
}

impl CatalogError {
    /// Create a parse error with a catalog path.
    #[must_use]
    pub fn parse(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Parse {
            path: path.into(),
            message: message.into(),
        }
    }

    /// Create a validation error with a catalog path.
    #[must_use]
    pub fn validation(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Validation {
            path: path.into(),
            message: message.into(),
        }
    }

    /// Catalog path associated with this error.
    #[must_use]
    pub fn path(&self) -> &str {
        match self {
            Self::Parse { path, .. } | Self::Validation { path, .. } => path,
        }
    }

    /// Human-readable detail without the path prefix.
    #[must_use]
    pub fn message(&self) -> &str {
        match self {
            Self::Parse { message, .. } | Self::Validation { message, .. } => message,
        }
    }
}

static BUILTIN_MODELS: OnceLock<Result<BuiltinModels, CatalogError>> = OnceLock::new();

/// Load the compiled-in built-in model catalog.
///
/// The catalog is parsed once and shared for the process lifetime. Failures are
/// typed and path-rich; this function never panics on invalid data.
///
/// # Errors
///
/// Returns a path-rich [`CatalogError`] if the compiled JSON is malformed or
/// any model fails required-field or catalog-key validation.
pub fn builtin_models() -> Result<&'static BuiltinModels, CatalogError> {
    match BUILTIN_MODELS.get_or_init(load_builtin_models) {
        Ok(models) => Ok(models),
        Err(error) => Err(error.clone()),
    }
}

fn load_builtin_models() -> Result<BuiltinModels, CatalogError> {
    let root: Value = serde_json::from_str(BUILTIN_MODELS_JSON).map_err(|error| {
        CatalogError::parse(BUILTIN_MODELS_PATH, format!("invalid JSON: {error}"))
    })?;
    parse_builtin_models_value(&root, BUILTIN_MODELS_PATH)
}

fn parse_builtin_models_value(
    root: &Value,
    root_path: &str,
) -> Result<BuiltinModels, CatalogError> {
    let providers = root
        .as_object()
        .ok_or_else(|| CatalogError::validation(root_path, "catalog root must be a JSON object"))?;

    let mut out = BuiltinModels::new();
    for (provider_id, provider_value) in providers {
        let provider_path = format!("{root_path}.{provider_id}");
        let models_obj = provider_value.as_object().ok_or_else(|| {
            CatalogError::validation(
                provider_path.clone(),
                "provider entry must be a JSON object of models",
            )
        })?;

        let mut models = BTreeMap::new();
        for (model_id, model_value) in models_obj {
            let model_path = format!("{provider_path}.{model_id}");
            let model = model_from_value(model_value, &model_path)?;
            if model.provider != *provider_id {
                return Err(CatalogError::validation(
                    format!("{model_path}.provider"),
                    format!(
                        "model.provider `{provider}` does not match catalog key `{provider_id}`",
                        provider = model.provider
                    ),
                ));
            }
            if model.id != *model_id {
                return Err(CatalogError::validation(
                    format!("{model_path}.id"),
                    format!(
                        "model.id `{id}` does not match catalog key `{model_id}`",
                        id = model.id
                    ),
                ));
            }
            models.insert(model_id.clone(), model);
        }
        out.insert(provider_id.clone(), models);
    }
    Ok(out)
}

/// Deserialize a model value and enforce required non-empty fields.
pub(crate) fn model_from_value(value: &Value, path: &str) -> Result<Model, CatalogError> {
    if !value.is_object() {
        return Err(CatalogError::validation(
            path,
            "model entry must be a JSON object",
        ));
    }

    let model: Model = serde_json::from_value(value.clone()).map_err(|error| {
        CatalogError::validation(path, format!("failed to decode model: {error}"))
    })?;
    validate_required_model_fields(&model, path)?;
    Ok(model)
}

pub(crate) fn validate_required_model_fields(
    model: &Model,
    path: &str,
) -> Result<(), CatalogError> {
    if model.id.trim().is_empty() {
        return Err(CatalogError::validation(
            format!("{path}.id"),
            "required field is empty",
        ));
    }
    if model.name.trim().is_empty() {
        return Err(CatalogError::validation(
            format!("{path}.name"),
            "required field is empty",
        ));
    }
    if model.api.trim().is_empty() {
        return Err(CatalogError::validation(
            format!("{path}.api"),
            "required field is empty",
        ));
    }
    if model.provider.trim().is_empty() {
        return Err(CatalogError::validation(
            format!("{path}.provider"),
            "required field is empty",
        ));
    }
    if model.input.is_empty() {
        return Err(CatalogError::validation(
            format!("{path}.input"),
            "required field must contain at least one modality",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModelCost, ModelInput};
    use serde_json::json;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn assert_float_eq(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= f64::EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    fn required<T>(value: Option<T>, path: &str) -> Result<T, CatalogError> {
        value.ok_or_else(|| CatalogError::validation(path, "test fixture value is missing"))
    }

    fn expected_error<T>(result: Result<T, CatalogError>) -> Result<CatalogError, CatalogError> {
        match result {
            Err(error) => Ok(error),
            Ok(_) => Err(CatalogError::validation(
                "test",
                "operation unexpectedly succeeded",
            )),
        }
    }

    #[test]
    fn builtin_catalog_parses_39_providers_and_1353_models() -> TestResult {
        let catalog = builtin_models()?;
        assert_eq!(catalog.len(), 39, "provider count");
        let model_count: usize = catalog.values().map(BTreeMap::len).sum();
        assert_eq!(model_count, 1353, "model count");
        assert!(
            !catalog.contains_key("radius"),
            "radius is intentionally absent: it is a dynamic OAuth/gateway provider, not a static catalog entry"
        );
        Ok(())
    }
    #[test]
    fn every_catalog_provider_has_a_native_registration() -> TestResult {
        use crate::providers::KnownProvider;

        let catalog = builtin_models()?;
        let mut missing = Vec::new();
        for provider_id in catalog.keys() {
            if KnownProvider::from_id(provider_id).is_none() {
                missing.push(provider_id.clone());
            }
        }
        assert!(
            missing.is_empty(),
            "catalog providers without native registration: {missing:?}"
        );
        Ok(())
    }
    #[test]
    fn explicit_cache_mode_models_carry_their_opt_in_flag() -> TestResult {
        let catalog = builtin_models()?;
        for id in ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"] {
            let model = required(catalog.get("openai").and_then(|models| models.get(id)), id)?;
            assert_eq!(model.api, "openai-responses", "{id} api");
            assert_eq!(
                model
                    .compat
                    .as_ref()
                    .and_then(|compat| compat.get("supportsExplicitPromptCacheMode"))
                    .and_then(Value::as_bool),
                Some(true),
                "{id} explicit cache flag",
            );
        }
        Ok(())
    }

    #[test]
    fn representative_builtin_fields_match_checked_in_payload() -> TestResult {
        let catalog = builtin_models()?;
        let gpt4 = required(
            catalog.get("openai").and_then(|models| models.get("gpt-4")),
            "openai.gpt-4",
        )?;
        assert_eq!(gpt4.id, "gpt-4");
        assert_eq!(gpt4.name, "GPT-4");
        assert_eq!(gpt4.api, "openai-responses");
        assert_eq!(gpt4.provider, "openai");
        assert_eq!(gpt4.base_url, "https://api.openai.com/v1");
        assert!(!gpt4.reasoning);
        assert_eq!(gpt4.input, vec![ModelInput::Text]);
        assert_eq!(gpt4.context_window, 8192);
        assert_eq!(gpt4.max_tokens, 8192);
        assert_eq!(
            gpt4.cost,
            ModelCost {
                input: 30.0,
                output: 60.0,
                cache_read: 0.0,
                cache_write: 0.0,
                tiers: None,
            }
        );

        let gemini = required(
            catalog
                .get("google")
                .and_then(|models| models.get("gemini-2.5-pro")),
            "google.gemini-2.5-pro",
        )?;
        assert_eq!(gemini.api, "google-generative-ai");
        assert_eq!(gemini.provider, "google");
        assert!(gemini.reasoning);
        assert_eq!(gemini.input, vec![ModelInput::Text, ModelInput::Image]);
        assert_eq!(gemini.context_window, 1_048_576);
        assert_eq!(gemini.max_tokens, 65_536);
        assert_float_eq(gemini.cost.input, 1.25);
        assert_float_eq(gemini.cost.output, 10.0);
        assert_float_eq(gemini.cost.cache_read, 0.125);

        let fable = required(
            catalog
                .get("amazon-bedrock")
                .and_then(|models| models.get("anthropic.claude-fable-5")),
            "amazon-bedrock.anthropic.claude-fable-5",
        )?;
        let thinking = required(fable.thinking_level_map.as_ref(), "thinkingLevelMap")?;
        assert!(
            thinking.contains_key(&crate::types::ModelThinkingLevel::Off),
            "off key must be present"
        );
        assert_eq!(
            thinking.get(&crate::types::ModelThinkingLevel::Off),
            Some(&None),
            "off is explicitly unsupported (JSON null)"
        );
        assert_eq!(
            thinking
                .get(&crate::types::ModelThinkingLevel::Max)
                .cloned()
                .flatten()
                .as_deref(),
            Some("max")
        );
        Ok(())
    }

    #[test]
    fn runtime_catalog_path_cannot_spawn_bun() -> TestResult {
        // Strip the test module so assertions cannot self-match their needles.
        fn production(src: &str) -> &str {
            src.split_once("#[cfg(test)]")
                .map_or(src, |(production, _tests)| production)
        }

        let mod_src = include_str!("mod.rs");
        let merge_src = include_str!("merge.rs");
        for (name, src) in [("mod.rs", mod_src), ("merge.rs", merge_src)] {
            let code = production(src);
            let process = ["std", "process"].join("::");
            let command = ["Command", "new"].join("::");
            let tokio_process = ["tokio", "process"].join("::");
            let bun_cmd = ["bun", "run"].join(" ");
            assert!(!code.contains(&process), "{name} must not use {process}");
            assert!(!code.contains(&command), "{name} must not spawn processes");
            assert!(
                !code.contains(&tokio_process),
                "{name} must not use {tokio_process}"
            );
            assert!(
                !code.contains("\"bun\"")
                    && !code.contains("'bun'")
                    && !code.contains(&bun_cmd)
                    && !code.contains("bun.exe"),
                "{name} must not reference a Bun runtime spawn"
            );
        }
        assert!(
            production(mod_src).contains("include_str!"),
            "catalog must load via include_str!"
        );
        // Loading must succeed from the compiled-in payload alone.
        let _catalog = builtin_models()?;
        Ok(())
    }

    #[test]
    fn parse_rejects_missing_required_fields_with_path() -> TestResult {
        let value = json!({
            "demo": {
                "broken": {
                    "id": "broken",
                    "name": "Broken",
                    "api": "",
                    "provider": "demo",
                    "baseUrl": "https://example.test",
                    "reasoning": false,
                    "input": ["text"],
                    "cost": {
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0
                    },
                    "contextWindow": 1,
                    "maxTokens": 1
                }
            }
        });
        let error = expected_error(parse_builtin_models_value(&value, "fixture.json"))?;
        assert_eq!(error.path(), "fixture.json.demo.broken.api");
        assert!(error.message().contains("empty"));
        Ok(())
    }

    #[test]
    fn parse_rejects_provider_and_model_key_mismatch() -> TestResult {
        let value = json!({
            "demo": {
                "broken": {
                    "id": "other-id",
                    "name": "Broken",
                    "api": "openai-completions",
                    "provider": "other-provider",
                    "baseUrl": "https://example.test",
                    "reasoning": false,
                    "input": ["text"],
                    "cost": {
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0
                    },
                    "contextWindow": 1,
                    "maxTokens": 1
                }
            }
        });
        let error = expected_error(parse_builtin_models_value(&value, "fixture.json"))?;
        assert_eq!(error.path(), "fixture.json.demo.broken.provider");

        let value = json!({
            "demo": {
                "broken": {
                    "id": "other-id",
                    "name": "Broken",
                    "api": "openai-completions",
                    "provider": "demo",
                    "baseUrl": "https://example.test",
                    "reasoning": false,
                    "input": ["text"],
                    "cost": {
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0
                    },
                    "contextWindow": 1,
                    "maxTokens": 1
                }
            }
        });
        let error = expected_error(parse_builtin_models_value(&value, "fixture.json"))?;
        assert_eq!(error.path(), "fixture.json.demo.broken.id");
        Ok(())
    }

    #[test]
    fn model_round_trip_keeps_unknown_fields() -> TestResult {
        let value = json!({
            "id": "x",
            "name": "X",
            "api": "openai-completions",
            "provider": "demo",
            "baseUrl": "https://example.test",
            "reasoning": false,
            "input": ["text"],
            "cost": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0
            },
            "contextWindow": 1024,
            "maxTokens": 256,
            "futureField": {"nested": true, "count": 2}
        });
        let model = model_from_value(&value, "model.x")?;
        assert_eq!(model.extra["futureField"]["nested"], json!(true));
        assert_eq!(model.extra["futureField"]["count"], json!(2));
        let encoded = serde_json::to_value(&model)?;
        assert_eq!(encoded["futureField"]["nested"], json!(true));
        Ok(())
    }
}
