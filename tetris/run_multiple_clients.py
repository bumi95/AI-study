import asyncio
import subprocess
import sys
import os

async def run_client(client_id: int):
    """테트리스 클라이언트 실행"""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # 파이썬 출력 버퍼링 비활성화
    
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "./snake-ai-pytorch-main/tetris/client_ml.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        print(f"클라이언트 {client_id} 시작됨 (PID: {process.pid})")
        
        async def read_stream(stream, prefix):
            while True:
                line = await stream.readline()
                if not line:
                    break
                print(f"{prefix}: {line.decode().strip()}")

        await asyncio.gather(
            read_stream(process.stdout, f"클라이언트 {client_id} (stdout)"),
            read_stream(process.stderr, f"클라이언트 {client_id} (stderr)")
        )
        
        await process.wait()
        print(f"클라이언트 {client_id} 종료됨 (반환 코드: {process.returncode})")
    except Exception as e:
        print(f"클라이언트 {client_id} 실행 중 오류 발생: {e}")

async def main():
    """여러 테트리스 클라이언트 동시 실행"""
    num_clients = 5
    clients = [run_client(i+1) for i in range(num_clients)]
    await asyncio.gather(*clients)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"메인 스크립트 실행 중 오류 발생: {e}")
