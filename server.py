from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pickle
import uuid
import shutil
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from qwen_code import SimpleQwen  


# try limiting the cache size (e.g., to 512MB) to force more frequent returns to the OS to help with mem availibility.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for React development

# Configuration
SESSIONS_DIR = Path("chat_sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

# In-memory store for active sessions (can be replaced with Redis or some other DB for persistence)
active_sessions: Dict[str, Any] = {}

# Initialize chat
coder = None
# Inject the initialized coder into the global namespace for routes to use
def initialize_coder():
    global coder
    # Initialize chat - This now only runs in the main process!
    coder = SimpleQwen(force_offline=True) 

class SessionManager:
    """Manages chat sessions including messages and uploaded files."""
    
    @staticmethod
    def create_session() -> str:
        """Create a new chat session with unique ID."""
        session_id = str(datetime.now().timestamp())#str(uuid.uuid4())
        session_path = SESSIONS_DIR / session_id
        session_path.mkdir(exist_ok=True)
        
        # Initialize session data
        session_data = {
            "messages": [{"role": "system", "content": "You are a helpful coding assistant."}],
            "files": []
        }
        
        # Save to disk
        SessionManager.save_session(session_id, session_data)
        
        return session_id
    
    @staticmethod
    def get_session_path(session_id: str) -> Path:
        """Get the directory path for a session."""
        return SESSIONS_DIR / session_id
    
    @staticmethod
    def save_session(session_id: str, session_data: Dict):
        """Save session data to pickle file."""
        session_path = SessionManager.get_session_path(session_id)
        session_path.mkdir(exist_ok=True)
        
        with open(session_path / "session.pkl", "wb") as f:
            pickle.dump(session_data, f)
    
    @staticmethod
    def load_session(session_id: str) -> Dict:
        """Load session data from pickle file."""
        session_path = SessionManager.get_session_path(session_id)
        pickle_path = session_path / "session.pkl"
        
        if not pickle_path.exists():
            return None
        
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    
    @staticmethod
    def list_sessions() -> List[Dict]:
        """List all available chat sessions."""
        sessions = []
        
        for session_dir in SESSIONS_DIR.iterdir():
            if session_dir.is_dir():
                session_data = SessionManager.load_session(session_dir.name)
                if session_data:
                    # Get preview from first user message or use default
                    preview = "New Conversation"
                    for msg in session_data["messages"]:
                        if msg["role"] == "user":
                            preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                            break
                    
                    sessions.append({
                        "id": session_dir.name,
                        "preview": preview,
                        "message_count": len([m for m in session_data["messages"] if m["role"] != "system"])
                    })
        
        return sessions
    
    @staticmethod
    def delete_session(session_id: str):
        """Delete a session and all its files."""
        session_path = SessionManager.get_session_path(session_id)
        if session_path.exists():
            shutil.rmtree(session_path)


# API Routes

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get list of all chat sessions."""
    sessions = SessionManager.list_sessions()
    return jsonify({"sessions": sessions})


@app.route('/api/sessions', methods=['POST'])
def create_session():
    """Create a new chat session."""
    session_id = SessionManager.create_session()
    return jsonify({"session_id": session_id})


@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Load a specific chat session."""
    global coder
    session_data = SessionManager.load_session(session_id)
    
    if not session_data:
        return jsonify({"error": "Session not found"}), 404
    else:
        coder.messages = session_data["messages"]
    
    return jsonify(session_data)


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a chat session."""
    SessionManager.delete_session(session_id)
    return jsonify({"success": True})


@app.route('/api/sessions/<session_id>/system-prompt', methods=['POST'])
def update_system_prompt(session_id):
    global coder
    """Update the system prompt for a session."""
    data = request.json
    system_prompt = data.get('content', '')
    
    session_data = SessionManager.load_session(session_id)
    if not session_data:
        return jsonify({"error": "Session not found"}), 404
    
    # Update system message (always first message)
    session_data["messages"][0] = {"role": "system", "content": system_prompt}
    SessionManager.save_session(session_id, session_data)

    coder.messages[0] = {"role": "system", "content": system_prompt}
    
    return jsonify({"success": True})


@app.route('/api/sessions/<session_id>/messages', methods=['POST'])
def send_message(session_id):
    global coder
    """Send a message and get AI response."""
    data = request.json
    user_message = data.get('content', '')
    
    session_data = SessionManager.load_session(session_id)
    if not session_data:
        return jsonify({"error": "Session not found"}), 404
    
    # Calculate token count
    input_token_count = coder.token_count(user_message)

    # Add user message
    session_data["messages"].append({"role": "user", "content": user_message, "token_count": input_token_count})

    # Get AI response
    response = coder.chat(user_message, file_contents=session_data["files"])
    assistant_response = f"{response}"
    
    # Calculate token count
    output_token_count = coder.token_count(assistant_response)

    session_data["messages"].append({"role": "assistant", "content": assistant_response, "token_count": output_token_count})
    
    SessionManager.save_session(session_id, session_data)

    #calculate total tokens
    total_tokens = 0
    for msg in session_data["messages"]:
        if "token_count" in msg:
            total_tokens += msg["token_count"]
    print(f"Total tokens in session {session_id}: {total_tokens}")

    return jsonify({
        "message": session_data["messages"][-1],
        "all_messages": session_data["messages"],
        "total_tokens": str(total_tokens)
    })


@app.route('/api/sessions/<session_id>/messages/<int:message_index>', methods=['PUT'])
def update_message(session_id, message_index):
    """Update a specific message."""
    global coder
    data = request.json
    new_content = data.get('content', '')
    
    session_data = SessionManager.load_session(session_id)
    if not session_data:
        return jsonify({"error": "Session not found"}), 404
    
    if message_index >= len(session_data["messages"]):
        return jsonify({"error": "Message index out of range"}), 400
    
    session_data["messages"][message_index]["content"] = new_content
    SessionManager.save_session(session_id, session_data)

    coder.messages = session_data["messages"]
    
    return jsonify({"success": True})


@app.route('/api/sessions/<session_id>/messages/<int:message_index>', methods=['DELETE'])
def delete_message(session_id, message_index):
    """Delete a specific message."""
    session_data = SessionManager.load_session(session_id)
    if not session_data:
        return jsonify({"error": "Session not found"}), 404
    
    # Don't allow deleting system message
    if message_index == 0:
        return jsonify({"error": "Cannot delete system message"}), 400
    
    if message_index >= len(session_data["messages"]):
        return jsonify({"error": "Message index out of range"}), 400
    
    del session_data["messages"][message_index]
    SessionManager.save_session(session_id, session_data)
    
    return jsonify({"success": True, "messages": session_data["messages"]})


@app.route('/api/sessions/<session_id>/files', methods=['POST'])
def upload_file(session_id):
    """Upload a file to the session."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check file extension
    allowed_extensions = {'.py', '.txt', '.html', '.js'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        return jsonify({"error": f"File type {file_ext} not allowed"}), 400
    
    session_data = SessionManager.load_session(session_id)
    if not session_data:
        return jsonify({"error": "Session not found"}), 404
    
    # Save file to session directory
    session_path = SessionManager.get_session_path(session_id)
    file_path = session_path / file.filename
    file.save(file_path)
    
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add to session files list
    file_info = {
        "filename": file.filename,
        "size": len(content),
        "content": content
    }
    session_data["files"].append(file_info)
    
    SessionManager.save_session(session_id, session_data)
    
    return jsonify({"file": file_info})


@app.route('/api/sessions/<session_id>/files/<filename>', methods=['DELETE'])
def delete_file(session_id, filename):
    """Delete a file from the session."""
    session_data = SessionManager.load_session(session_id)
    if not session_data:
        return jsonify({"error": "Session not found"}), 404
    
    # Remove from files list
    session_data["files"] = [f for f in session_data["files"] if f["filename"] != filename]
    
    # Delete physical file
    session_path = SessionManager.get_session_path(session_id)
    file_path = session_path / filename
    if file_path.exists():
        file_path.unlink()
    
    SessionManager.save_session(session_id, session_data)
    
    return jsonify({"success": True})


@app.route('/')
def serve_index():
    """Serve the React app."""
    return send_from_directory('static', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

@app.route('/lib/<path:filename>')
def serve_lib_files(filename):
    """
    Serves files requested from the /lib/ path.
    Uses send_from_directory for security (prevents path traversal).
    """
    LIB_DIR = os.path.join(app.root_path, 'lib')
    try:
        return send_from_directory(LIB_DIR, filename)
    except FileNotFoundError:
        return "Resource not found.", 404


if __name__ == '__main__':
    import socket
    initialize_coder() # Ensure coder is initialized in the main process
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    port = 5000
    print(f"Flask server is running at http://{local_ip}:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)  # use_reloader=False to prevent double initialization