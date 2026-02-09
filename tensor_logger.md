# Shape Log

| Layer                                                              | Type               | Input shape     | Output shape    |
| :----------------------------------------------------------------- | :----------------- | :-------------- | :-------------- |
| `tokenizer.categorical_embeddings.0`                               | Embedding          | (20,)           | (20, 128)       |
| `tokenizer.categorical_embeddings.1`                               | Embedding          | (20,)           | (20, 128)       |
| `tokenizer.categorical_embeddings.2`                               | Embedding          | (20,)           | (20, 128)       |
| `tokenizer.categorical_embeddings.3`                               | Embedding          | (20,)           | (20, 128)       |
| `tokenizer.categorical_embeddings.4`                               | Embedding          | (20,)           | (20, 128)       |
| `tokenizer.categorical_embeddings.5`                               | Embedding          | (20,)           | (20, 128)       |
| `tokenizer.categorical_embeddings.6`                               | Embedding          | (20,)           | (20, 128)       |
| `tokenizer.categorical_embeddings.7`                               | Embedding          | (20,)           | (20, 128)       |
| `tokenizer.categorical_embeddings.8`                               | Embedding          | (20,)           | (20, 128)       |
| `tokenizer.categorical_embeddings.9`                               | Embedding          | (20,)           | (20, 128)       |
| `tokenizer.categorical_embeddings.10`                              | Embedding          | (20,)           | (20, 128)       |
| `tokenizer.categorical_embeddings.11`                              | Embedding          | (20,)           | (20, 128)       |
| `tokenizer.categorical_embeddings.12`                              | Embedding          | (20,)           | (20, 128)       |
| `tokenizer.categorical_embeddings.13`                              | Embedding          | (20,)           | (20, 128)       |
| `tokenizer.categorical_embeddings.14`                              | Embedding          | (20,)           | (20, 128)       |
| `tokenizer.categorical_embeddings.15`                              | Embedding          | (20,)           | (20, 128)       |
| `tokenizer.categorical_embeddings.16`                              | Embedding          | (20,)           | (20, 128)       |
| `tokenizer.embedding_dropout`                                      | Dropout            | (20, 17, 128)   | (20, 17, 128)   |
| `tokenizer.continuous_embedding.projection`                        | Linear             | (20, 37, 128)   | (20, 37, 128)   |
| `tokenizer.continuous_embedding.gate`                              | Linear             | (20, 37, 128)   | (20, 37, 128)   |
| `tokenizer.continuous_embedding`                                   | FourierFeatures    | (20, 37)        | (20, 37, 128)   |
| `tokenizer`                                                        | FeatureTokenizer   | (4, 5, 17)      | (4, 5, 54, 128) |
| `invoice_encoder.layers.0.layer_norm_1`                            | LayerNorm          | (20, 54, 128)   | (20, 54, 128)   |
| `invoice_encoder.layers.0.query_key_value`                         | Linear             | (20, 54, 128)   | (20, 54, 384)   |
| `invoice_encoder.layers.0.output_projection`                       | Linear             | (20, 54, 128)   | (20, 54, 128)   |
| `invoice_encoder.layers.0.drop_path_1`                             | StochasticDepth    | (20, 54, 128)   | (20, 54, 128)   |
| `invoice_encoder.layers.0.layer_norm_2`                            | LayerNorm          | (20, 54, 128)   | (20, 54, 128)   |
| `invoice_encoder.layers.0.feed_forward_network.gate_projection`    | Linear             | (20, 54, 128)   | (20, 54, 512)   |
| `invoice_encoder.layers.0.feed_forward_network.up_projection`      | Linear             | (20, 54, 128)   | (20, 54, 512)   |
| `invoice_encoder.layers.0.feed_forward_network.output_projection`  | Linear             | (20, 54, 512)   | (20, 54, 128)   |
| `invoice_encoder.layers.0.feed_forward_network.dropout_layer`      | Dropout            | (20, 54, 128)   | (20, 54, 128)   |
| `invoice_encoder.layers.0.feed_forward_network`                    | SwiGLU             | (20, 54, 128)   | (20, 54, 128)   |
| `invoice_encoder.layers.0.drop_path_2`                             | StochasticDepth    | (20, 54, 128)   | (20, 54, 128)   |
| `invoice_encoder.layers.0`                                         | TransformerBlock   | (20, 54, 128)   | (20, 54, 128)   |
| `invoice_encoder.pool.0`                                           | LayerNorm          | (20, 128)       | (20, 128)       |
| `invoice_encoder.pool.1`                                           | Linear             | (20, 128)       | (20, 128)       |
| `invoice_encoder`                                                  | InvoiceEncoder     | (4, 5, 54, 128) | (4, 5, 128)     |
| `sequence_encoder.layers.0.layer_norm_1`                           | LayerNorm          | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.0.query_key_value`                        | Linear             | (4, 5, 128)     | (4, 5, 384)     |
| `sequence_encoder.rotary_positional_embedding`                     | RoPE               | (4, 4, 5, 32)   | <class 'tuple'> |
| `sequence_encoder.layers.0.output_projection`                      | Linear             | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.0.drop_path_1`                            | StochasticDepth    | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.0.layer_norm_2`                           | LayerNorm          | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.0.feed_forward_network.gate_projection`   | Linear             | (4, 5, 128)     | (4, 5, 512)     |
| `sequence_encoder.layers.0.feed_forward_network.up_projection`     | Linear             | (4, 5, 128)     | (4, 5, 512)     |
| `sequence_encoder.layers.0.feed_forward_network.output_projection` | Linear             | (4, 5, 512)     | (4, 5, 128)     |
| `sequence_encoder.layers.0.feed_forward_network.dropout_layer`     | Dropout            | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.0.feed_forward_network`                   | SwiGLU             | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.0.drop_path_2`                            | StochasticDepth    | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.0`                                        | TransformerBlock   | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.1.layer_norm_1`                           | LayerNorm          | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.1.query_key_value`                        | Linear             | (4, 5, 128)     | (4, 5, 384)     |
| `sequence_encoder.rotary_positional_embedding`                     | RoPE               | (4, 4, 5, 32)   | <class 'tuple'> |
| `sequence_encoder.layers.1.output_projection`                      | Linear             | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.1.drop_path_1`                            | StochasticDepth    | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.1.layer_norm_2`                           | LayerNorm          | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.1.feed_forward_network.gate_projection`   | Linear             | (4, 5, 128)     | (4, 5, 512)     |
| `sequence_encoder.layers.1.feed_forward_network.up_projection`     | Linear             | (4, 5, 128)     | (4, 5, 512)     |
| `sequence_encoder.layers.1.feed_forward_network.output_projection` | Linear             | (4, 5, 512)     | (4, 5, 128)     |
| `sequence_encoder.layers.1.feed_forward_network.dropout_layer`     | Dropout            | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.1.feed_forward_network`                   | SwiGLU             | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.1.drop_path_2`                            | StochasticDepth    | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.1`                                        | TransformerBlock   | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.2.layer_norm_1`                           | LayerNorm          | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.2.query_key_value`                        | Linear             | (4, 5, 128)     | (4, 5, 384)     |
| `sequence_encoder.rotary_positional_embedding`                     | RoPE               | (4, 4, 5, 32)   | <class 'tuple'> |
| `sequence_encoder.layers.2.output_projection`                      | Linear             | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.2.drop_path_1`                            | StochasticDepth    | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.2.layer_norm_2`                           | LayerNorm          | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.2.feed_forward_network.gate_projection`   | Linear             | (4, 5, 128)     | (4, 5, 512)     |
| `sequence_encoder.layers.2.feed_forward_network.up_projection`     | Linear             | (4, 5, 128)     | (4, 5, 512)     |
| `sequence_encoder.layers.2.feed_forward_network.output_projection` | Linear             | (4, 5, 512)     | (4, 5, 128)     |
| `sequence_encoder.layers.2.feed_forward_network.dropout_layer`     | Dropout            | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.2.feed_forward_network`                   | SwiGLU             | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.2.drop_path_2`                            | StochasticDepth    | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layers.2`                                        | TransformerBlock   | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder.layer_norm`                                      | LayerNorm          | (4, 5, 128)     | (4, 5, 128)     |
| `sequence_encoder`                                                 | SequenceEncoder    | (4, 5, 128)     | <class 'tuple'> |
| `temporal_attention.attention`                                     | MultiheadAttention | (4, 1, 128)     | <class 'tuple'> |
| `temporal_attention.dropout_layer`                                 | Dropout            | (4, 128)        | (4, 128)        |
| `temporal_attention.gated_residual_network.fully_connected_1`      | Linear             | (4, 128)        | (4, 128)        |
| `temporal_attention.gated_residual_network.fully_connected_2`      | Linear             | (4, 128)        | (4, 128)        |
| `temporal_attention.gated_residual_network.dropout_layer`          | Dropout            | (4, 128)        | (4, 128)        |
| `temporal_attention.gated_residual_network.gate_layer`             | Linear             | (4, 128)        | (4, 128)        |
| `temporal_attention.gated_residual_network.layer_norm`             | LayerNorm          | (4, 128)        | (4, 128)        |
| `temporal_attention.gated_residual_network`                        | GRN                | (4, 128)        | (4, 128)        |
| `temporal_attention`                                               | CrossAttention     | (4, 128)        | <class 'tuple'> |
| `head_days.gated_residual_network_1.fully_connected_1`             | Linear             | (4, 384)        | (4, 128)        |
| `head_days.gated_residual_network_1.fully_connected_2`             | Linear             | (4, 128)        | (4, 128)        |
| `head_days.gated_residual_network_1.dropout_layer`                 | Dropout            | (4, 128)        | (4, 128)        |
| `head_days.gated_residual_network_1.gate_layer`                    | Linear             | (4, 128)        | (4, 128)        |
| `head_days.gated_residual_network_1.skip_connection`               | Linear             | (4, 384)        | (4, 128)        |
| `head_days.gated_residual_network_1.layer_norm`                    | LayerNorm          | (4, 128)        | (4, 128)        |
| `head_days.gated_residual_network_1`                               | GRN                | (4, 384)        | (4, 128)        |
| `head_days.gated_residual_network_2.fully_connected_1`             | Linear             | (4, 128)        | (4, 64)         |
| `head_days.gated_residual_network_2.fully_connected_2`             | Linear             | (4, 64)         | (4, 64)         |
| `head_days.gated_residual_network_2.dropout_layer`                 | Dropout            | (4, 64)         | (4, 64)         |
| `head_days.gated_residual_network_2.gate_layer`                    | Linear             | (4, 64)         | (4, 64)         |
| `head_days.gated_residual_network_2.skip_connection`               | Linear             | (4, 128)        | (4, 64)         |
| `head_days.gated_residual_network_2.layer_norm`                    | LayerNorm          | (4, 64)         | (4, 64)         |
| `head_days.gated_residual_network_2`                               | GRN                | (4, 128)        | (4, 64)         |
| `head_days.output_layer`                                           | Linear             | (4, 64)         | (4, 1)          |
| `head_days`                                                        | PredictionHead     | (4, 384)        | (4, 1)          |

**Records:** 103