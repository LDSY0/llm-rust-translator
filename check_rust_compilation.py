import subprocess
import tempfile
import os

def check_rust_compilation(code: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "temp_check.rs")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            result = subprocess.run(
                ["rustc", "--emit=metadata", file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                text=True
            )
            success = result.returncode == 0
            return success, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Compilation timed out."
        except Exception as e:
            return False, "", str(e)


# rust_code = """
# fn main() {
#     prin!("Hello, world!");
# }
# """

# success, stdout, stderr = check_rust_compilation(rust_code)
# if success:
#     print("✅ 编译成功")
# else:
#     print("❌ 编译失败")
#     print("错误信息：")
#     print(stderr)
