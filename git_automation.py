import subprocess
import sys
import os


def install_chocolatey():
    """Chocolatey 설치"""
    try:
        subprocess.check_call(["choco", "--version"])
        print("Chocolatey가 이미 설치되어 있습니다.")
    except subprocess.CalledProcessError:
        print("Chocolatey가 설치되지 않았습니다. 설치를 시작합니다.")
        # PowerShell 경로를 명시적으로 지정하여 Chocolatey 설치
        powershell_path = "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"
        subprocess.check_call(
            [powershell_path, '-Command',
             'Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12; iex ((New-Object System.Net.WebClient).DownloadString("https://chocolatey.org/install.ps1"))']
        )


def install_git():
    """Git 설치"""
    try:
        subprocess.check_call(["git", "--version"])
        print("Git이 이미 설치되어 있습니다.")
    except subprocess.CalledProcessError:
        print("Git이 설치되지 않았습니다. 설치를 시작합니다.")
        subprocess.check_call(["choco", "install", "git", "-y"])
        print("Git이 성공적으로 설치되었습니다.")


def check_git_installed():
    """Git 설치 확인"""
    try:
        subprocess.check_call(["git", "--version"])
        print("Git이 정상적으로 설치되었습니다.")
    except subprocess.CalledProcessError:
        print("Git 설치에 실패했습니다.")
        sys.exit(1)


def initialize_git_repo():
    """Git 리포지토리 초기화 및 첫 커밋"""
    if not os.path.exists(".git"):
        subprocess.check_call(["git", "init"])
        subprocess.check_call(['git', 'add', '.'])
        subprocess.check_call(['git', 'commit', '-m', '"Initial commit"'])
        print("Git 리포지토리가 초기화되었습니다.")
    else:
        print("이미 Git 리포지토리가 초기화되어 있습니다.")


def push_to_github(repo_url):
    """GitHub 리모트 저장소 추가 및 푸시"""
    subprocess.check_call(['git', 'branch', '-M', 'main'])
    subprocess.check_call(['git', 'remote', 'add', 'origin', repo_url])
    subprocess.check_call(['git', 'push', '-u', 'origin', 'main'])
    print("GitHub에 코드가 푸시되었습니다.")


if __name__ == "__main__":
    # 1. Chocolatey 설치
    install_chocolatey()

    # 2. Git 설치
    install_git()

    # 3. Git 설치 확인
    check_git_installed()

    # 4. Git 리포지토리 초기화 및 첫 커밋
    initialize_git_repo()

    # 5. GitHub 리모트 저장소에 푸시
    repo_url = "https://github.com/park3hho/llm_pretrain.git"  # 본인의 리포지토리 URL로 변경
    push_to_github(repo_url)
