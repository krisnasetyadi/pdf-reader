# chat_parser.py
"""
Parser for chat export files (WhatsApp TXT, Teams JSON, Slack JSON)
Primary focus: WhatsApp TXT format
"""

import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from collections import Counter

from models import ChatMessage, ChatPlatform

logger = logging.getLogger(__name__)


class ChatParser:
    """Parser for various chat export formats"""
    
    # WhatsApp date/time patterns (handles multiple formats)
    WHATSAPP_PATTERNS = [
        # Format: [YYYY-MM-DD HH:MM:SS] Sender: Message (custom format)
        r'\[(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\]\s*([^:]+):\s*(.*)',
        # Format: [DD/MM/YYYY, HH:MM:SS] Sender: Message
        r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?)\]\s*([^:]+):\s*(.*)',
        # Format: DD/MM/YYYY, HH:MM - Sender: Message
        r'(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*([^:]+):\s*(.*)',
        # Format: MM/DD/YY, HH:MM AM/PM - Sender: Message (US format)
        r'(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}\s*(?:AM|PM)?)\s*-\s*([^:]+):\s*(.*)',
    ]
    
    # System message patterns to skip
    SYSTEM_PATTERNS = [
        r'Messages and calls are end-to-end encrypted',
        r'created group',
        r'added you',
        r'removed you',
        r'left the group',
        r'changed the subject',
        r'changed this group',
        r'security code changed',
        r'<Media omitted>',
        r'missed voice call',
        r'missed video call',
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.WHATSAPP_PATTERNS]
        self.system_patterns = [re.compile(p, re.IGNORECASE) for p in self.SYSTEM_PATTERNS]
    
    def parse_whatsapp(self, file_path: str, encoding: str = 'utf-8') -> Tuple[List[ChatMessage], Dict[str, Any]]:
        """
        Parse WhatsApp TXT export file
        
        Returns:
            Tuple of (list of ChatMessage, metadata dict)
        """
        messages = []
        participants = set()
        current_message = None
        
        try:
            # Try different encodings
            content = None
            for enc in [encoding, 'utf-8-sig', 'utf-16', 'latin-1']:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise ValueError(f"Could not decode file with any supported encoding")
            
            lines = content.split('\n')
            logger.info(f"📱 Parsing WhatsApp file with {len(lines)} lines")
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Skip system messages
                if self._is_system_message(line):
                    continue
                
                # Try to parse as new message
                parsed = self._parse_whatsapp_line(line)
                
                if parsed:
                    # Save previous message if exists
                    if current_message:
                        messages.append(current_message)
                    
                    timestamp, sender, content = parsed
                    participants.add(sender)
                    
                    current_message = ChatMessage(
                        message_id=f"wa_{line_num}",
                        sender=sender.strip(),
                        timestamp=timestamp,
                        content=content.strip(),
                        platform=ChatPlatform.WHATSAPP,
                        raw_line=line
                    )
                elif current_message:
                    # Continuation of previous message (multiline)
                    current_message.content += f"\n{line}"
            
            # Don't forget the last message
            if current_message:
                messages.append(current_message)
            
            # Build metadata
            metadata = self._build_metadata(messages, participants, file_path)
            
            logger.info(f"✅ Parsed {len(messages)} messages from {len(participants)} participants")
            return messages, metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to parse WhatsApp file: {e}")
            raise
    
    def _parse_whatsapp_line(self, line: str) -> Optional[Tuple[datetime, str, str]]:
        """Try to parse a line as a WhatsApp message"""
        for pattern in self.compiled_patterns:
            match = pattern.match(line)
            if match:
                groups = match.groups()
                date_str = groups[0]
                time_str = groups[1]
                sender = groups[2]
                content = groups[3]
                
                # Parse timestamp
                timestamp = self._parse_timestamp(date_str, time_str)
                if timestamp:
                    return timestamp, sender, content
        
        return None
    
    def _parse_timestamp(self, date_str: str, time_str: str) -> Optional[datetime]:
        """Parse date and time strings into datetime object"""
        # Clean up time string
        time_str = time_str.strip()
        
        # Common date formats
        date_formats = [
            '%Y-%m-%d',              # YYYY-MM-DD (ISO format)
            '%d/%m/%Y', '%d/%m/%y',  # DD/MM/YYYY or DD/MM/YY
            '%m/%d/%Y', '%m/%d/%y',  # MM/DD/YYYY or MM/DD/YY (US)
            '%Y/%m/%d',              # YYYY/MM/DD
        ]
        
        # Common time formats
        time_formats = [
            '%H:%M:%S', '%H:%M',           # 24-hour
            '%I:%M:%S %p', '%I:%M %p',     # 12-hour with AM/PM
            '%I:%M%p',                      # 12-hour without space
        ]
        
        for date_fmt in date_formats:
            for time_fmt in time_formats:
                try:
                    dt_str = f"{date_str} {time_str}"
                    fmt = f"{date_fmt} {time_fmt}"
                    return datetime.strptime(dt_str, fmt)
                except ValueError:
                    continue
        
        # If all formats fail, try just the date
        for date_fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, date_fmt)
                # Add a default time
                return dt.replace(hour=0, minute=0, second=0)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse timestamp: {date_str} {time_str}")
        return None
    
    def _is_system_message(self, line: str) -> bool:
        """Check if line is a system message that should be skipped"""
        for pattern in self.system_patterns:
            if pattern.search(line):
                return True
        return False
    
    def _build_metadata(self, messages: List[ChatMessage], participants: set, file_path: str) -> Dict[str, Any]:
        """Build metadata dictionary from parsed messages"""
        if not messages:
            return {
                "file_name": Path(file_path).name,
                "message_count": 0,
                "participants": [],
                "date_range": None
            }
        
        timestamps = [m.timestamp for m in messages if m.timestamp]
        
        return {
            "file_name": Path(file_path).name,
            "message_count": len(messages),
            "participants": sorted(list(participants)),
            "date_range": {
                "start": min(timestamps).isoformat() if timestamps else None,
                "end": max(timestamps).isoformat() if timestamps else None
            }
        }
    
    def chunk_messages_by_conversation(
        self, 
        messages: List[ChatMessage], 
        chunk_size: int = 10,
        overlap: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Chunk messages into conversation segments for better context
        
        Args:
            messages: List of ChatMessage objects
            chunk_size: Number of messages per chunk
            overlap: Number of overlapping messages between chunks
            
        Returns:
            List of chunks, each containing messages and metadata
        """
        if not messages:
            return []
        
        chunks = []
        step = chunk_size - overlap
        
        for i in range(0, len(messages), step):
            chunk_messages = messages[i:i + chunk_size]
            
            if not chunk_messages:
                continue
            
            # Build chunk text
            chunk_text = self._format_chunk_text(chunk_messages)
            
            # Get participants in this chunk
            chunk_participants = list(set(m.sender for m in chunk_messages))
            
            # Get time range
            timestamps = [m.timestamp for m in chunk_messages if m.timestamp]
            
            chunk = {
                "text": chunk_text,
                "messages": chunk_messages,
                "message_count": len(chunk_messages),
                "participants": chunk_participants,
                "start_index": i,
                "end_index": i + len(chunk_messages) - 1,
                "time_range": {
                    "start": min(timestamps).isoformat() if timestamps else None,
                    "end": max(timestamps).isoformat() if timestamps else None
                }
            }
            
            chunks.append(chunk)
        
        logger.info(f"📦 Created {len(chunks)} conversation chunks from {len(messages)} messages")
        return chunks
    
    def _format_chunk_text(self, messages: List[ChatMessage]) -> str:
        """Format messages into searchable text"""
        lines = []
        for msg in messages:
            timestamp_str = msg.timestamp.strftime("%Y-%m-%d %H:%M") if msg.timestamp else ""
            lines.append(f"[{timestamp_str}] {msg.sender}: {msg.content}")
        return "\n".join(lines)
    
    def extract_keywords(
        self, 
        messages: List[ChatMessage], 
        top_n: int = 20,
        min_word_length: int = 3
    ) -> List[str]:
        """
        Extract important keywords from chat messages using frequency analysis
        
        Args:
            messages: List of ChatMessage objects
            top_n: Number of top keywords to return
            min_word_length: Minimum length of words to consider
            
        Returns:
            List of important keywords sorted by relevance
        """
        # Indonesian stopwords (common words to ignore)
        stopwords = {
            'yang', 'dan', 'dari', 'untuk', 'ini', 'itu', 'pada', 'adalah', 'dengan', 'tidak',
            'ada', 'akan', 'sudah', 'juga', 'atau', 'bisa', 'saya', 'kita', 'kami', 'mereka',
            'dia', 'pak', 'bu', 'mas', 'mbak', 'bang', 'kak', 'bos', 'gan', 'sis', 'guys',
            'the', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall',
            'this', 'that', 'these', 'those', 'of', 'to', 'for', 'in', 'on', 'at', 'by', 'with',
            'a', 'an', 'and', 'or', 'but', 'if', 'then', 'than', 'so', 'very', 'too', 'also'
        }
        
        # Collect all words from messages
        word_freq = Counter()
        
        for msg in messages:
            # Tokenize content (simple word splitting)
            words = re.findall(r'\b\w+\b', msg.content.lower())
            
            for word in words:
                # Filter by length and stopwords
                if (len(word) >= min_word_length and 
                    word not in stopwords and 
                    not word.isdigit()):
                    word_freq[word] += 1
        
        # Get participants (important for name-based queries)
        participants = set()
        for msg in messages:
            # Extract first names from sender
            name_parts = msg.sender.lower().split()
            participants.update(name_parts)
        
        # Boost participant names (they're often important for queries)
        for name in participants:
            if name in word_freq:
                word_freq[name] *= 2  # Double the frequency for names
        
        # Get top keywords
        top_keywords = [word for word, _ in word_freq.most_common(top_n)]
        
        logger.info(f"🔑 Extracted {len(top_keywords)} keywords: {top_keywords[:10]}...")
        return top_keywords


# Convenience function
def parse_chat_file(file_path: str, platform: str = "whatsapp") -> Tuple[List[ChatMessage], Dict[str, Any]]:
    """
    Parse chat file based on platform
    
    Args:
        file_path: Path to chat export file
        platform: Chat platform (whatsapp, teams, slack)
        
    Returns:
        Tuple of (messages, metadata)
    """
    parser = ChatParser()
    
    if platform.lower() == "whatsapp":
        return parser.parse_whatsapp(file_path)
    else:
        raise ValueError(f"Unsupported platform: {platform}. Currently only 'whatsapp' is supported.")
