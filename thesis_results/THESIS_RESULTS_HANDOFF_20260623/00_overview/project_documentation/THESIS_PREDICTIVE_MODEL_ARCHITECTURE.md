# Verified Predictive Model Architecture

This description was verified by constructing all five model families with the frozen Phase 1 architecture configuration in TensorFlow 2.10.1 and inspecting the instantiated Keras layer graphs.

Common constants are:

- masking value: 2;
- AP first recurrent layer: bidirectional LSTM, 5 units per direction, sequence output enabled;
- AP second recurrent layer: unidirectional LSTM, 5 units;
- AP dense activation: SELU;
- sequence-branch convolutions: two Conv1D layers, five filters each, same padding, linear activation, He-normal initialization;
- dropout rate: 0.5;
- output: one sigmoid neuron;
- L2 coefficient on AP dense layers: 0.0.

## AP

Ordered inputs:

1. amino-acid AP;
2. dipeptide AP;
3. tripeptide AP.

Each input is processed independently:

```text
Input
-> Masking(mask_value=2)
-> Bidirectional LSTM(5 units per direction, return_sequences=True)
-> LSTM(5 units, unidirectional)
-> Dense(128, SELU)
-> Dropout(0.5)
```

The three 128-dimensional outputs are concatenated to 384 dimensions and passed to `Dense(1, sigmoid)`.

The frozen AP value `num_cells=64` controls the AP dense width through `dense=2*num_cells=128`. It does not control either AP recurrent layer; those remain fixed at 5 units.

## SP

Ordered input: one physicochemical sequence tensor.

```text
Input
-> Masking(mask_value=2)
-> Conv1D(5 filters, kernel=6, same padding, linear)
-> Conv1D(5 filters, kernel=6, same padding, linear)
-> Bidirectional LSTM(32 units per direction)
-> Dropout(0.5)
-> Dense(1, sigmoid)
```

The bidirectional recurrent output has width 64.

## AP_SP

Ordered inputs:

1. amino-acid AP;
2. dipeptide AP;
3. tripeptide AP;
4. physicochemical SP tensor.

Each AP branch uses:

```text
Masking
-> Bidirectional LSTM(5 per direction, sequence output)
-> LSTM(5)
-> Dense(96, SELU)
-> Dropout(0.5)
```

The SP branch uses:

```text
Masking
-> Conv1D(5, kernel=8, same, linear)
-> Conv1D(5, kernel=8, same, linear)
-> Bidirectional LSTM(48 per direction)
-> Dropout(0.5)
```

The three 96-dimensional AP outputs and 96-dimensional SP output are concatenated to 384 dimensions and passed to `Dense(1, sigmoid)`.

## TSNE_SP

Ordered input: one standalone t-SNE tensor.

```text
Input
-> Masking(mask_value=2)
-> Conv1D(5 filters, kernel=6, same padding, linear)
-> Conv1D(5 filters, kernel=6, same padding, linear)
-> Bidirectional LSTM(48 units per direction)
-> Dropout(0.5)
-> Dense(1, sigmoid)
```

The recurrent output has width 96. The implemented input orientation is `(3,24)`, so Conv1D treats three positions as the temporal dimension and 24 values as channels. Whether this orientation was intended by the original authors needs confirmation.

## TSNE_AP_SP

Ordered inputs:

1. amino-acid AP;
2. dipeptide AP;
3. tripeptide AP;
4. transposed t-SNE tensor.

Each AP branch uses:

```text
Masking
-> Bidirectional LSTM(5 per direction, sequence output)
-> LSTM(5)
-> Dense(128, SELU)
-> Dropout(0.5)
```

The t-SNE branch uses:

```text
Masking
-> Conv1D(5, kernel=6, same, linear)
-> Conv1D(5, kernel=6, same, linear)
-> Bidirectional LSTM(64 per direction)
-> Dropout(0.5)
```

The three 128-dimensional AP outputs and 128-dimensional t-SNE output are concatenated to 512 dimensions and passed to `Dense(1, sigmoid)`.

## Masking caveat

AP branches pass the mask directly into mask-aware recurrent layers. SP and t-SNE branches place masking before Conv1D, but Keras Conv1D reports `supports_masking=False`. The model graph therefore contains the masking layer, but the mask is not propagated through convolution to the later bidirectional LSTM.

## Frozen architecture interpretation

| Family | Frozen `num_cells` | Actual interpretation |
|---|---:|---|
| AP | 64 | AP dense width is 128; AP recurrent units remain 5 and 5 |
| SP | 32 | 32 LSTM units in each direction |
| AP_SP | 48 | SP BiLSTM has 48 units per direction; each AP dense layer has width 96 |
| TSNE_SP | 48 | 48 LSTM units in each direction |
| TSNE_AP_SP | 64 | t-SNE BiLSTM has 64 units per direction; each AP dense layer has width 128 |

Thus, a table headed only "Recurrent cells" is misleading for AP and the hybrid models. The selected architecture parameter should instead be named `h` or "selected architecture size", followed by separate columns for AP recurrent units, sequence-branch recurrent units, and AP dense width.
