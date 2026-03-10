# 📁 Data

## Training Data

The training data is publicly available. Download it from:
[Link to your dataset]

### Structure expected

```
data/
└── train/
    ├── 0_very_negative/
    │   ├── img_001.jpg
    │   └── ...
    ├── 1_negative/
    ├── 2_neutral/
    ├── 3_positive/
    └── 4_very_positive/
```

## Test Data (Secret)

The test set is **never shared**. It is stored encrypted as a GitHub Actions secret and only accessed during automated evaluation.

### Format

```python
# test_data.pt  — shape: (N, 3, 224, 224), dtype: float32, values in [0, 1]
# test_labels.pt — shape: (N,),            dtype: int64,   values in {0,1,2,3,4}
test_data   = torch.load("test_data.pt")
test_labels = torch.load("test_labels.pt")
```

## Encoding the test set for GitHub Secrets

Run once to encode your test data:

```bash
python -c "
import base64
with open('test_data.pt', 'rb') as f:
    print(base64.b64encode(f.read()).decode())
" > test_data_b64.txt

python -c "
import base64
with open('test_labels.pt', 'rb') as f:
    print(base64.b64encode(f.read()).decode())
" > test_labels_b64.txt
```

Then add the contents as GitHub Secrets named:
- `TEST_DATA_B64`
- `TEST_LABELS_B64`
