use crate::program::Program;
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
        serde_wasm_bindgen::to_value(&self.inner.manifest()).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = query)]
    pub fn query(
        &self,
        major: Option<u32>,
        type_: Option<u32>,
        subtype: Option<u32>,
        label_contains: Option<String>,
    ) -> Result<JsValue, JsValue> {
        let space = crate::space::Space::new(&self.inner);
        let results = space.query(major, type_, subtype, label_contains.as_deref());
        serde_wasm_bindgen::to_value(&results).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
