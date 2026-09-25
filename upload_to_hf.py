"""
Script tự động đẩy Model đã fine-tune và FAISS Index lên Hugging Face Hub.
Lưu trữ vĩnh viễn trên Cloud - Chạy 1 lần, lưu mãi mãi, không lo mất file.
"""

import sys
from pathlib import Path
from huggingface_hub import HfApi, login

def main():
    print("=" * 60)
    print(" HƯỚNG DẪN ĐẨY MODEL LÊN HUGGING FACE HUB")
    print("=" * 60)
    
    # 1. Nhập Token
    hf_token = input("Nhập Hugging Face Access Token (quyền Write): ").strip()
    if not hf_token:
        print("Lỗi: Token không được để trống!")
        return
        
    try:
        login(token=hf_token)
        print(" Đăng nhập Hugging Face thành công!")
    except Exception as e:
        print(f" Đăng nhập thất bại: {e}")
        return

    api = HfApi()
    user_info = api.whoami()
    username = user_info["name"]
    print(f"Tài khoản hiện tại: {username}")

    # 2. Tên repo trên Hugging Face
    default_repo = "rag-t5-nlp-project"
    repo_name_input = input(f"Nhập tên Repo muốn tạo [mặc định: {default_repo}]: ").strip()
    repo_name = repo_name_input if repo_name_input else default_repo
    repo_id = f"{username}/{repo_name}"

    is_private_input = input("Bạn muốn để Repo ở chế độ Private (chỉ mình bạn thấy) không? (y/n) [mặc định: y]: ").strip().lower()
    is_private = False if is_private_input == "n" else True

    print(f"\n Đang tạo/kết nối repo: {repo_id} (Private: {is_private})...")
    api.create_repo(repo_id=repo_id, repo_type="model", private=is_private, exist_ok=True)

    # 3. Upload thư mục generator_finetuned
    generator_dir = Path("generator_finetuned")
    if not generator_dir.exists():
        print(f" Không tìm thấy thư mục {generator_dir}!")
        return

    print("\n[1/3] Đang tải lên các file mô hình T5 đã fine-tune (khoảng 890MB)...")
    api.upload_folder(
        folder_path=str(generator_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    print(" Upload mô hình T5 thành công!")

    # 4. Upload FAISS index và chunk metadata
    faiss_path = Path("data/faiss.index")
    meta_path = Path("data/chunk_metadata.jsonl")

    if faiss_path.exists():
        print("\n[2/3] Đang tải lên file FAISS index...")
        api.upload_file(
            path_or_fileobj=str(faiss_path),
            path_in_repo="faiss.index",
            repo_id=repo_id,
            repo_type="model",
        )
        print(" Upload FAISS index thành công!")

    if meta_path.exists():
        print("\n[3/3] Đang tải lên file chunk_metadata.jsonl...")
        api.upload_file(
            path_or_fileobj=str(meta_path),
            path_in_repo="chunk_metadata.jsonl",
            repo_id=repo_id,
            repo_type="model",
        )
        print(" Upload metadata thành công!")

    print("\n" + "=" * 60)
    print(f" HOÀN TẤT! Model và Index của bạn đã được lưu vĩnh viễn tại:")
    print(f" https://huggingface.co/{repo_id}")
    print("=" * 60)

if __name__ == "__main__":
    main()
