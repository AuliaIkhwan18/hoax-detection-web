import os
import gdown

# Link model dari Google Drive
models = {
    "bert_hoax_classifier.pth":
    "https://drive.google.com/uc?id=1UswJ0EJRFeRjbWuxjvKutOUu53x4cA1c",

    "svm_model.pkl":
    "https://drive.google.com/uc?id=1yCVjntV_CW7yi7oJvQGWWSAb4VtSjf6D",

    "tfidf_vectorizer.pkl":
    "https://drive.google.com/uc?id=1jnCHIHtZCPFFmXd4lKSM3cqK21P7Jv9V"
}

# Download jika file belum ada
for filename, url in models.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        gdown.download(url, filename, quiet=False)
    else:
        print(f"{filename} already exists.")