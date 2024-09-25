# 843m
# 96 prunable layers
exclude_layers_12 = [
    "module.language_model.encoder.layers.23.mlp.dense_h_to_4h",
    "module.language_model.encoder.layers.23.mlp.dense_4h_to_h",
    "module.language_model.encoder.layers.7.mlp.dense_h_to_4h",
    "module.language_model.encoder.layers.0.mlp.dense_h_to_4h",
    "module.language_model.encoder.layers.22.mlp.dense_h_to_4h",
    "module.language_model.encoder.layers.8.self_attention.query_key_value",
    "module.language_model.encoder.layers.7.mlp.dense_4h_to_h",
    "module.language_model.encoder.layers.19.mlp.dense_h_to_4h",
    "module.language_model.encoder.layers.23.self_attention.query_key_value",
    "module.language_model.encoder.layers.21.mlp.dense_h_to_4h",
    "module.language_model.encoder.layers.20.mlp.dense_h_to_4h",
    "module.language_model.encoder.layers.0.mlp.dense_4h_to_h",
]

# 8b
# 128 prunable layers
exclude_layers_13 = [
    "module.language_model.encoder.layers.31.mlp.dense_h_to_4h",
    "module.language_model.encoder.layers.31.mlp.dense_4h_to_h",
    "module.language_model.encoder.layers.30.self_attention.query_key_value",
    "module.language_model.encoder.layers.30.mlp.dense_h_to_4h",
    "module.language_model.encoder.layers.29.mlp.dense_h_to_4h",
    "module.language_model.encoder.layers.28.mlp.dense_h_to_4h",
    "module.language_model.encoder.layers.30.mlp.dense_4h_to_h",
    "module.language_model.encoder.layers.27.mlp.dense_h_to_4h",
    "module.language_model.encoder.layers.22.self_attention.query_key_value",
    "module.language_model.encoder.layers.25.mlp.dense_h_to_4h",
    "module.language_model.encoder.layers.26.self_attention.query_key_value",
    "module.language_model.encoder.layers.26.mlp.dense_h_to_4h",
    "module.language_model.encoder.layers.31.self_attention.query_key_value",
]

exclude_layers_1000 = ["mlp.dense_4h_to_h"]