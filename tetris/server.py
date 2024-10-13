import asyncio
import websockets
import json
import logging
from typing import Dict, Any

# 상수 정의
MAX_PLAYERS = 5
HOST = "localhost"
PORT = 8765

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TetrisServer:
    """테트리스 멀티플레이어 게임 서버 클래스"""

    def __init__(self, host: str, port: int):
        """서버 초기화"""
        self.host = host
        self.port = port
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.game_states: Dict[str, Any] = {}
        logging.info("테트리스 서버 초기화 완료")

    async def start(self):
        """서버 시작"""
        server = await websockets.serve(self.handle_client, self.host, self.port)
        logging.info(f"서버 시작: {self.host}:{self.port}")
        try:
            await server.wait_closed()
        except asyncio.CancelledError:
            logging.info("서버 종료 중...")
            server.close()
            await server.wait_closed()
            logging.info("서버가 정상적으로 종료되었습니다.")

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str) -> None:
        """클라이언트 연결 처리"""
        client_id = str(id(websocket))
        try:
            await self.register(websocket, client_id)
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosedOK:
            logging.info(f"클라이언트가 정상적으로 연결 종료: {client_id}")
        except websockets.exceptions.ConnectionClosedError as e:
            logging.error(f"클라이언트 연결 오류: {client_id}, 오류: {e}")
        except Exception as e:
            logging.error(f"예기치 않은 오류 발생: {client_id}, 오류: {e}")
        finally:
            await self.unregister(client_id)

    async def register(self, websocket: websockets.WebSocketServerProtocol, client_id: str) -> bool:
        """새 클라이언트 등록"""
        if len(self.clients) >= MAX_PLAYERS:
            await websocket.send(json.dumps({"type": "error", "message": "서버가 가득 찼습니다."}))
            logging.warning(f"최대 플레이어 수 초과로 연결 거부됨: {websocket.remote_address}")
            return False
        
        self.clients[client_id] = websocket
        self.game_states[client_id] = {"score": 0, "board": []}
        await websocket.send(json.dumps({"type": "connection_success", "player_id": client_id}))
        logging.info(f"새 클라이언트 등록: {client_id}, 현재 클라이언트 수: {len(self.clients)}")
        await self.notify_clients()
        return True

    async def unregister(self, client_id: str) -> None:
        """클라이언트 연결 해제"""
        if client_id in self.clients:
            del self.clients[client_id]
        if client_id in self.game_states:
            del self.game_states[client_id]
        logging.info(f"클라이언트 연결 해제: {client_id}, 현재 클라이언트 수: {len(self.clients)}")
        await self.notify_clients()

    async def notify_clients(self) -> None:
        """모든 클라이언트에게 업데이트 전송"""
        if not self.clients:
            return
        message = json.dumps({
            "type": "update",
            "clients": list(self.clients.keys()),
            "game_states": self.game_states
        })
        await asyncio.gather(*(self.send_to_client(client, message) for client in list(self.clients.values())), return_exceptions=True)

    async def send_to_client(self, client: websockets.WebSocketServerProtocol, message: str) -> None:
        """클라이언트에게 메시지 전송"""
        try:
            await client.send(message)
        except websockets.exceptions.ConnectionClosed:
            logging.warning(f"메시지 전송 실패: 클라이언트 연결이 닫힘")
        except Exception as e:
            logging.error(f"메시지 전송 중 오류 발생: {e}")

    async def handle_message(self, websocket: websockets.WebSocketServerProtocol, message: str) -> None:
        """클라이언트로부터 받은 메시지 처리"""
        try:
            data = json.loads(message)
            client_id = str(id(websocket))
            if data["type"] == "update_state":
                self.game_states[client_id] = data["state"]
                await self.notify_clients()
            elif data["type"] == "insert_lines":
                await self.send_lines_to_others(client_id, data["num_lines"])
        except json.JSONDecodeError:
            logging.error(f"잘못된 JSON 형식: {message}")
        except KeyError as e:
            logging.error(f"필수 키 누락: {e}")
        except Exception as e:
            logging.error(f"메시지 처리 중 오류 발생: {e}")

    async def send_lines_to_others(self, sender_id: str, num_lines: int) -> None:
        """다른 플레이어들에게 라인 삽입 메시지 전송"""
        message = json.dumps({
            "type": "insert_lines",
            "num_lines": num_lines
        })
        tasks = [self.send_to_client(client, message) 
                 for cid, client in list(self.clients.items()) if cid != sender_id]
        await asyncio.gather(*tasks, return_exceptions=True)
        logging.info(f"플레이어 {sender_id}가 {num_lines}줄을 다른 플레이어들에게 보냄")

async def main() -> None:
    """메인 서버 실행 함수"""
    server = TetrisServer(HOST, PORT)
    await server.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("서버가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logging.error(f"서버 실행 중 예기치 않은 오류 발생: {e}", exc_info=True)
