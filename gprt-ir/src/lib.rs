use proc_macro2::TokenStream;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RtProgram {
    pub array_indices: HashMap<String, usize>,
    pub value_indices: HashMap<String, usize>,
    pub any_hit_tokens: Option<TokenStream>,
    pub closest_hit_tokens: Option<TokenStream>,
    // TODO: pub miss_tokens: Option<TokenStream>,
    pub geom_type: String,
}