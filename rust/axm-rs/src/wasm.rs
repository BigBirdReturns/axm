use crate::program::Program;
use crate::space::Space;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmProgram {
    inner: Program,
}

#[wasm_bindgen]
impl WasmProgram {
    #[wasm_bindgen(constructor)]
    pub fn from_bytes(data: &[u8]) -> Result<WasmProgram, JsValue> {
        console_error_panic_hook::set_once();
        Program::load_zip_bytes(data)
            .map(|inner| WasmProgram { inner })
            .map_err(|e| JsValue::from_str(&e))
    }

    #[wasm_bindgen(js_name = manifest)]
    pub fn manifest_json(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.inner.manifest())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = query)]
    pub fn query(
        &self,
        major: Option<u32>,
        type_: Option<u32>,
        subtype: Option<u32>,
        label_contains: Option<String>,
        value_gt: Option<f64>,
        value_lt: Option<f64>,
    ) -> Result<JsValue, JsValue> {
        let space = Space::new(&self.inner);
        let results = space.query(
            major,
            type_,
            subtype,
            label_contains.as_deref(),
            value_gt,
            value_lt,
        );
        serde_wasm_bindgen::to_value(&results).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = neighbors)]
    pub fn neighbors(&self, node_id: String, radius: f64) -> Result<JsValue, JsValue> {
        let space = Space::new(&self.inner);
        let results = space.neighbors(&node_id, radius, [1.0, 0.5, 0.3, 0.1]);
        serde_wasm_bindgen::to_value(&results).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
