import sqlite3
import json
import uuid
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation history with persistent storage using SQLite.
    Supports configurable memory limits and automatic cleanup.
    """
    
    def __init__(
        self, 
        db_path: str = "conversation_memory.db",
        max_messages: int = 10
    ):
        """
        Initialize the conversation memory system.
        
        Args:
            db_path: Path to SQLite database file
            max_messages: Maximum number of messages to keep per session
        """
        self.db_path = Path(db_path)
        self.max_messages = max_messages
        self._init_database()
        logger.info(f"ConversationMemory initialized with db_path={db_path}, max_messages={max_messages}")
    
    def _init_database(self):
        """Create the database schema if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_id 
                    ON conversations(session_id, timestamp)
                """)
                conn.commit()
                logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def create_session(self) -> str:
        """
        Create a new conversation session.
        
        Returns:
            A unique session ID (UUID4)
        """
        session_id = str(uuid.uuid4())
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str,
        metadata: Optional[Dict] = None
    ):
        """
        Add a message to the conversation history.
        
        Args:
            session_id: The session identifier
            role: Message role ('user' or 'assistant')
            content: The message content
            metadata: Optional metadata (e.g., retrieved context info)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO conversations (session_id, role, content, metadata)
                    VALUES (?, ?, ?, ?)
                """, (session_id, role, content, json.dumps(metadata) if metadata else None))
                conn.commit()
                
            # Cleanup old messages if limit exceeded
            self._cleanup_old_messages(session_id)
            logger.debug(f"Added {role} message to session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to add message to session {session_id}: {e}")
            raise
    
    def _cleanup_old_messages(self, session_id: str):
        """Remove old messages beyond the max_messages limit."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count current messages
                cursor.execute("""
                    SELECT COUNT(*) FROM conversations WHERE session_id = ?
                """, (session_id,))
                count = cursor.fetchone()[0]
                
                # Delete oldest messages if limit exceeded
                if count > self.max_messages:
                    to_delete = count - self.max_messages
                    cursor.execute("""
                        DELETE FROM conversations
                        WHERE id IN (
                            SELECT id FROM conversations
                            WHERE session_id = ?
                            ORDER BY timestamp ASC
                            LIMIT ?
                        )
                    """, (session_id, to_delete))
                    conn.commit()
                    logger.debug(f"Cleaned up {to_delete} old messages from session {session_id}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup messages for session {session_id}: {e}")
    
    def get_history(
        self, 
        session_id: str, 
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Retrieve conversation history for a session.
        
        Args:
            session_id: The session identifier
            limit: Optional limit on number of messages to return
            
        Returns:
            List of messages in format [{"role": "user", "content": "..."}, ...]
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if limit:
                    cursor.execute("""
                        SELECT role, content, timestamp 
                        FROM conversations
                        WHERE session_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (session_id, limit))
                else:
                    cursor.execute("""
                        SELECT role, content, timestamp 
                        FROM conversations
                        WHERE session_id = ?
                        ORDER BY timestamp ASC
                    """, (session_id,))
                
                rows = cursor.fetchall()
                
                # Reverse if we used DESC order with limit
                if limit:
                    rows = rows[::-1]
                
                history = [{"role": row[0], "content": row[1]} for row in rows]
                logger.debug(f"Retrieved {len(history)} messages for session {session_id}")
                return history
                
        except Exception as e:
            logger.error(f"Failed to get history for session {session_id}: {e}")
            return []
    
    def clear_session(self, session_id: str):
        """
        Clear all messages for a specific session.
        
        Args:
            session_id: The session identifier
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM conversations WHERE session_id = ?
                """, (session_id,))
                conn.commit()
                logger.info(f"Cleared session: {session_id}")
                
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            raise
    
    def get_all_sessions(self) -> List[str]:
        """
        Get all active session IDs.
        
        Returns:
            List of session IDs
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT session_id FROM conversations
                    ORDER BY MAX(timestamp) DESC
                """)
                sessions = [row[0] for row in cursor.fetchall()]
                logger.debug(f"Found {len(sessions)} active sessions")
                return sessions
                
        except Exception as e:
            logger.error(f"Failed to get sessions: {e}")
            return []
    
    def delete_old_sessions(self, days: int = 30):
        """
        Delete sessions older than specified days.
        
        Args:
            days: Number of days to keep
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM conversations
                    WHERE timestamp < datetime('now', '-' || ? || ' days')
                """, (days,))
                deleted = cursor.rowcount
                conn.commit()
                logger.info(f"Deleted {deleted} messages from sessions older than {days} days")
                
        except Exception as e:
            logger.error(f"Failed to delete old sessions: {e}")
            raise
    
    def get_session_info(self, session_id: str) -> Dict:
        """
        Get metadata about a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Dictionary with session statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        COUNT(*) as message_count,
                        MIN(timestamp) as first_message,
                        MAX(timestamp) as last_message
                    FROM conversations
                    WHERE session_id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                if row and row[0] > 0:
                    return {
                        "session_id": session_id,
                        "message_count": row[0],
                        "first_message": row[1],
                        "last_message": row[2]
                    }
                return {"session_id": session_id, "message_count": 0}
                
        except Exception as e:
            logger.error(f"Failed to get session info for {session_id}: {e}")
            return {"session_id": session_id, "error": str(e)}
        