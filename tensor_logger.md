# Shape Log

| Layer                                                              | Type               | Input shape        | Output shape       |
| :----------------------------------------------------------------- | :----------------- | :----------------- | :----------------- |
| `tokenizer.categorical_embeddings.0`                               | Embedding          | (13056,)           | (13056, 128)       |
| `tokenizer.categorical_embeddings.1`                               | Embedding          | (13056,)           | (13056, 128)       |
| `tokenizer.categorical_embeddings.2`                               | Embedding          | (13056,)           | (13056, 128)       |
| `tokenizer.categorical_embeddings.3`                               | Embedding          | (13056,)           | (13056, 128)       |
| `tokenizer.categorical_embeddings.4`                               | Embedding          | (13056,)           | (13056, 128)       |
| `tokenizer.categorical_embeddings.5`                               | Embedding          | (13056,)           | (13056, 128)       |
| `tokenizer.categorical_embeddings.6`                               | Embedding          | (13056,)           | (13056, 128)       |
| `tokenizer.categorical_embeddings.7`                               | Embedding          | (13056,)           | (13056, 128)       |
| `tokenizer.categorical_embeddings.8`                               | Embedding          | (13056,)           | (13056, 128)       |
| `tokenizer.categorical_embeddings.9`                               | Embedding          | (13056,)           | (13056, 128)       |
| `tokenizer.categorical_embeddings.10`                              | Embedding          | (13056,)           | (13056, 128)       |
| `tokenizer.categorical_embeddings.11`                              | Embedding          | (13056,)           | (13056, 128)       |
| `tokenizer.categorical_embeddings.12`                              | Embedding          | (13056,)           | (13056, 128)       |
| `tokenizer.categorical_embeddings.13`                              | Embedding          | (13056,)           | (13056, 128)       |
| `tokenizer.categorical_embeddings.14`                              | Embedding          | (13056,)           | (13056, 128)       |
| `tokenizer.categorical_embeddings.15`                              | Embedding          | (13056,)           | (13056, 128)       |
| `tokenizer.categorical_embeddings.16`                              | Embedding          | (13056,)           | (13056, 128)       |
| `tokenizer.embedding_dropout`                                      | Dropout            | (13056, 17, 128)   | (13056, 17, 128)   |
| `tokenizer.continuous_embedding.projection`                        | Linear             | (13056, 37, 128)   | (13056, 37, 128)   |
| `tokenizer.continuous_embedding.gate`                              | Linear             | (13056, 37, 128)   | (13056, 37, 128)   |
| `tokenizer.continuous_embedding`                                   | FourierFeatures    | (13056, 37)        | (13056, 37, 128)   |
| `tokenizer`                                                        | FeatureTokenizer   | (256, 51, 17)      | (256, 51, 54, 128) |
| `invoice_encoder.layers.0.layer_norm_1`                            | LayerNorm          | (13056, 54, 128)   | (13056, 54, 128)   |
| `invoice_encoder.layers.0.query_key_value`                         | Linear             | (13056, 54, 128)   | (13056, 54, 384)   |
| `invoice_encoder.layers.0.output_projection`                       | Linear             | (13056, 54, 128)   | (13056, 54, 128)   |
| `invoice_encoder.layers.0.drop_path_1`                             | StochasticDepth    | (13056, 54, 128)   | (13056, 54, 128)   |
| `invoice_encoder.layers.0.layer_norm_2`                            | LayerNorm          | (13056, 54, 128)   | (13056, 54, 128)   |
| `invoice_encoder.layers.0.feed_forward_network.gate_projection`    | Linear             | (13056, 54, 128)   | (13056, 54, 512)   |
| `invoice_encoder.layers.0.feed_forward_network.up_projection`      | Linear             | (13056, 54, 128)   | (13056, 54, 512)   |
| `invoice_encoder.layers.0.feed_forward_network.output_projection`  | Linear             | (13056, 54, 512)   | (13056, 54, 128)   |
| `invoice_encoder.layers.0.feed_forward_network.dropout_layer`      | Dropout            | (13056, 54, 128)   | (13056, 54, 128)   |
| `invoice_encoder.layers.0.feed_forward_network`                    | SwiGLU             | (13056, 54, 128)   | (13056, 54, 128)   |
| `invoice_encoder.layers.0.drop_path_2`                             | StochasticDepth    | (13056, 54, 128)   | (13056, 54, 128)   |
| `invoice_encoder.layers.0`                                         | TransformerBlock   | (13056, 54, 128)   | (13056, 54, 128)   |
| `invoice_encoder.pool.0`                                           | LayerNorm          | (13056, 128)       | (13056, 128)       |
| `invoice_encoder.pool.1`                                           | Linear             | (13056, 128)       | (13056, 128)       |
| `invoice_encoder`                                                  | InvoiceEncoder     | (256, 51, 54, 128) | (256, 51, 128)     |
| `sequence_encoder.layers.0.layer_norm_1`                           | LayerNorm          | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.0.query_key_value`                        | Linear             | (256, 51, 128)     | (256, 51, 384)     |
| `sequence_encoder.rotary_positional_embedding`                     | RoPE               | (256, 4, 51, 32)   | <class 'tuple'>    |
| `sequence_encoder.layers.0.output_projection`                      | Linear             | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.0.drop_path_1`                            | StochasticDepth    | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.0.layer_norm_2`                           | LayerNorm          | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.0.feed_forward_network.gate_projection`   | Linear             | (256, 51, 128)     | (256, 51, 512)     |
| `sequence_encoder.layers.0.feed_forward_network.up_projection`     | Linear             | (256, 51, 128)     | (256, 51, 512)     |
| `sequence_encoder.layers.0.feed_forward_network.output_projection` | Linear             | (256, 51, 512)     | (256, 51, 128)     |
| `sequence_encoder.layers.0.feed_forward_network.dropout_layer`     | Dropout            | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.0.feed_forward_network`                   | SwiGLU             | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.0.drop_path_2`                            | StochasticDepth    | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.0`                                        | TransformerBlock   | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.1.layer_norm_1`                           | LayerNorm          | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.1.query_key_value`                        | Linear             | (256, 51, 128)     | (256, 51, 384)     |
| `sequence_encoder.rotary_positional_embedding`                     | RoPE               | (256, 4, 51, 32)   | <class 'tuple'>    |
| `sequence_encoder.layers.1.output_projection`                      | Linear             | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.1.drop_path_1`                            | StochasticDepth    | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.1.layer_norm_2`                           | LayerNorm          | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.1.feed_forward_network.gate_projection`   | Linear             | (256, 51, 128)     | (256, 51, 512)     |
| `sequence_encoder.layers.1.feed_forward_network.up_projection`     | Linear             | (256, 51, 128)     | (256, 51, 512)     |
| `sequence_encoder.layers.1.feed_forward_network.output_projection` | Linear             | (256, 51, 512)     | (256, 51, 128)     |
| `sequence_encoder.layers.1.feed_forward_network.dropout_layer`     | Dropout            | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.1.feed_forward_network`                   | SwiGLU             | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.1.drop_path_2`                            | StochasticDepth    | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.1`                                        | TransformerBlock   | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.2.layer_norm_1`                           | LayerNorm          | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.2.query_key_value`                        | Linear             | (256, 51, 128)     | (256, 51, 384)     |
| `sequence_encoder.rotary_positional_embedding`                     | RoPE               | (256, 4, 51, 32)   | <class 'tuple'>    |
| `sequence_encoder.layers.2.output_projection`                      | Linear             | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.2.drop_path_1`                            | StochasticDepth    | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.2.layer_norm_2`                           | LayerNorm          | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.2.feed_forward_network.gate_projection`   | Linear             | (256, 51, 128)     | (256, 51, 512)     |
| `sequence_encoder.layers.2.feed_forward_network.up_projection`     | Linear             | (256, 51, 128)     | (256, 51, 512)     |
| `sequence_encoder.layers.2.feed_forward_network.output_projection` | Linear             | (256, 51, 512)     | (256, 51, 128)     |
| `sequence_encoder.layers.2.feed_forward_network.dropout_layer`     | Dropout            | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.2.feed_forward_network`                   | SwiGLU             | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.2.drop_path_2`                            | StochasticDepth    | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layers.2`                                        | TransformerBlock   | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder.layer_norm`                                      | LayerNorm          | (256, 51, 128)     | (256, 51, 128)     |
| `sequence_encoder`                                                 | SequenceEncoder    | (256, 51, 128)     | <class 'tuple'>    |
| `temporal_attention.attention`                                     | MultiheadAttention | (256, 1, 128)      | <class 'tuple'>    |
| `temporal_attention.dropout_layer`                                 | Dropout            | (256, 128)         | (256, 128)         |
| `temporal_attention.gated_residual_network.fully_connected_1`      | Linear             | (256, 128)         | (256, 128)         |
| `temporal_attention.gated_residual_network.fully_connected_2`      | Linear             | (256, 128)         | (256, 128)         |
| `temporal_attention.gated_residual_network.dropout_layer`          | Dropout            | (256, 128)         | (256, 128)         |
| `temporal_attention.gated_residual_network.gate_layer`             | Linear             | (256, 128)         | (256, 128)         |
| `temporal_attention.gated_residual_network.layer_norm`             | LayerNorm          | (256, 128)         | (256, 128)         |
| `temporal_attention.gated_residual_network`                        | GRN                | (256, 128)         | (256, 128)         |
| `temporal_attention`                                               | CrossAttention     | (256, 128)         | <class 'tuple'>    |
| `head_days.gated_residual_network_1.fully_connected_1`             | Linear             | (256, 384)         | (256, 128)         |
| `head_days.gated_residual_network_1.fully_connected_2`             | Linear             | (256, 128)         | (256, 128)         |
| `head_days.gated_residual_network_1.dropout_layer`                 | Dropout            | (256, 128)         | (256, 128)         |
| `head_days.gated_residual_network_1.gate_layer`                    | Linear             | (256, 128)         | (256, 128)         |
| `head_days.gated_residual_network_1.skip_connection`               | Linear             | (256, 384)         | (256, 128)         |
| `head_days.gated_residual_network_1.layer_norm`                    | LayerNorm          | (256, 128)         | (256, 128)         |
| `head_days.gated_residual_network_1`                               | GRN                | (256, 384)         | (256, 128)         |
| `head_days.gated_residual_network_2.fully_connected_1`             | Linear             | (256, 128)         | (256, 64)          |
| `head_days.gated_residual_network_2.fully_connected_2`             | Linear             | (256, 64)          | (256, 64)          |
| `head_days.gated_residual_network_2.dropout_layer`                 | Dropout            | (256, 64)          | (256, 64)          |
| `head_days.gated_residual_network_2.gate_layer`                    | Linear             | (256, 64)          | (256, 64)          |
| `head_days.gated_residual_network_2.skip_connection`               | Linear             | (256, 128)         | (256, 64)          |
| `head_days.gated_residual_network_2.layer_norm`                    | LayerNorm          | (256, 64)          | (256, 64)          |
| `head_days.gated_residual_network_2`                               | GRN                | (256, 128)         | (256, 64)          |
| `head_days.output_layer`                                           | Linear             | (256, 64)          | (256, 1)           |
| `head_days`                                                        | PredictionHead     | (256, 384)         | (256, 1)           |

**Records:** 103