"""
Python-C# 통신 모듈
다양한 통신 방식을 지원하는 통합 모듈
"""

import json
import sys
import os
import socket
import threading
import time
from typing import Dict, Any, Optional
import pickle
import struct
import tempfile
import mmap
import queue
import multiprocessing as mp

class CommunicationManager:
    """Python-C# 통신 관리자"""
    
    def __init__(self, method='json_file'):
        self.method = method
        self.server_socket = None
        self.client_socket = None
        self.is_running = False
        
    def send_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 전송 및 응답 수신"""
        if self.method == 'json_file':
            return self._json_file_communication(data)
        elif self.method == 'socket':
            return self._socket_communication(data)
        elif self.method == 'shared_memory':
            return self._shared_memory_communication(data)
        elif self.method == 'pipe':
            return self._pipe_communication(data)
        else:
            raise ValueError(f"Unsupported communication method: {self.method}")
    
    def _json_file_communication(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """JSON 파일 기반 통신 (현재 방식)"""
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            request_file = f.name
        
        try:
            # C# 프로세스 실행
            import subprocess
            result = subprocess.run([
                'python', 'python_inference.py', request_file
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return {'status': 'success', 'output': result.stdout}
            else:
                return {'status': 'error', 'error': result.stderr}
                
        finally:
            # 임시 파일 정리
            if os.path.exists(request_file):
                os.unlink(request_file)
    
    def _socket_communication(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """소켓 기반 통신 (더 빠름)"""
        try:
            # 소켓 연결
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', 8888))
            
            # 데이터 전송
            message = json.dumps(data).encode('utf-8')
            sock.send(struct.pack('I', len(message)))
            sock.send(message)
            
            # 응답 수신
            length_data = sock.recv(4)
            if len(length_data) < 4:
                return {'status': 'error', 'error': 'Connection closed'}
            
            length = struct.unpack('I', length_data)[0]
            response_data = sock.recv(length)
            response = json.loads(response_data.decode('utf-8'))
            
            sock.close()
            return response
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _shared_memory_communication(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """공유 메모리 기반 통신 (가장 빠름)"""
        try:
            # 공유 메모리 생성
            shm_name = 'rosc_communication'
            shm_size = 1024 * 1024  # 1MB
            
            # 데이터 직렬화
            data_bytes = pickle.dumps(data)
            
            # 공유 메모리에 쓰기
            with mmap.mmap(-1, shm_size, tagname=shm_name) as mm:
                mm.write(struct.pack('I', len(data_bytes)))
                mm.write(data_bytes)
            
            # C# 프로세스에 신호 전송
            # (실제 구현에서는 네임드 파이프나 이벤트 사용)
            
            # 응답 대기 및 읽기
            time.sleep(0.1)  # 처리 시간 대기
            
            with mmap.mmap(-1, shm_size, tagname=shm_name) as mm:
                length_data = mm.read(4)
                if len(length_data) < 4:
                    return {'status': 'error', 'error': 'No response'}
                
                length = struct.unpack('I', length_data)[0]
                response_data = mm.read(length)
                response = pickle.loads(response_data)
                
            return response
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _pipe_communication(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """네임드 파이프 기반 통신"""
        try:
            pipe_name = r'\\.\pipe\rosc_pipe'
            
            # 파이프 연결
            with open(pipe_name, 'w') as pipe:
                json.dump(data, pipe)
            
            # 응답 읽기
            with open(pipe_name, 'r') as pipe:
                response = json.load(pipe)
                
            return response
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

class SocketServer:
    """소켓 서버 (Python 측)"""
    
    def __init__(self, port=8888):
        self.port = port
        self.server_socket = None
        self.is_running = False
        
    def start(self):
        """서버 시작"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('localhost', self.port))
        self.server_socket.listen(1)
        self.is_running = True
        
        print(f"Python 서버 시작: 포트 {self.port}")
        
        while self.is_running:
            try:
                client_socket, addr = self.server_socket.accept()
                threading.Thread(target=self._handle_client, args=(client_socket,)).start()
            except Exception as e:
                print(f"서버 오류: {e}")
                break
    
    def _handle_client(self, client_socket):
        """클라이언트 요청 처리"""
        try:
            # 요청 수신
            length_data = client_socket.recv(4)
            if len(length_data) < 4:
                return
            
            length = struct.unpack('I', length_data)[0]
            request_data = client_socket.recv(length)
            request = json.loads(request_data.decode('utf-8'))
            
            # 요청 처리
            response = self._process_request(request)
            
            # 응답 전송
            response_data = json.dumps(response).encode('utf-8')
            client_socket.send(struct.pack('I', len(response_data)))
            client_socket.send(response_data)
            
        except Exception as e:
            error_response = {'status': 'error', 'error': str(e)}
            response_data = json.dumps(error_response).encode('utf-8')
            client_socket.send(struct.pack('I', len(response_data)))
            client_socket.send(response_data)
        finally:
            client_socket.close()
    
    def _process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """요청 처리 (실제 추론 로직)"""
        # 여기에 실제 추론 로직 구현
        return {'status': 'success', 'result': 'processed'}

if __name__ == '__main__':
    # 통신 방식 테스트
    comm = CommunicationManager('json_file')
    result = comm.send_data({'action': 'test', 'data': 'hello'})
    print(f"결과: {result}")
