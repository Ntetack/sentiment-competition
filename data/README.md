# 📁 Data

## sample_submission.csv

Template CSV provided to participants. All labels set to `0` — participants replace with their predictions.

## Secret Labels

Your secret labels file must be a CSV:

```csv
image_id,label
img_0001,3
img_0002,0
img_0003,5
```

### How to store it as a GitHub Secret

Since labels are just a small CSV (text), storing in GitHub Secrets is perfectly fine:

```bash
# Copy the content of your secret_labels.csv
cat secret_labels.csv
```

Then:
1. Go to your repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **"New repository secret"**
3. Name: `SECRET_LABELS_CSV`
4. Value: paste the full CSV content (including header)
5. Click **"Add secret"**

That's it. The pipeline reads it with:
```yaml
echo "$SECRET_LABELS_CSV" > /tmp/secret_labels.csv
```
