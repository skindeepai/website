"""
PLGL Zero-Prompt Social Media

Like TikTok's algorithm but for AI-generated content.
No prompting needed - just swipe to train, then enjoy infinite personalized content.

Key innovation: Removes the friction of prompting from generative AI!
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class ContentItem:
    """Single piece of generated content"""
    content_type: str  # 'video', 'music', 'image', 'story'
    latent: torch.Tensor
    content: any  # The actual generated content
    engagement_score: float = 0.0
    view_duration: float = 0.0
    user_action: Optional[str] = None  # 'like', 'skip', 'share', 'save'


class ZeroPromptFeed:
    """
    Infinite personalized content feed without any prompting
    
    Just like TikTok's ForYou page, but every piece of content
    is AI-generated specifically for you!
    """
    
    def __init__(self, generators: Dict[str, any], latent_dim=512):
        self.generators = generators  # Different generators for each content type
        self.latent_dim = latent_dim
        self.preference_models = {}
        self.engagement_history = deque(maxlen=1000)
        self.current_session = []
        
    def start_new_session(self, user_id: str, content_type: str = 'mixed'):
        """
        Start a new content session for user
        No prompts needed - just start swiping!
        """
        print(f"\n🎬 Starting zero-prompt feed for {user_id}")
        print("Just swipe! The AI learns what you like...\n")
        
        if user_id not in self.preference_models:
            self.preference_models[user_id] = {
                ctype: self._create_preference_model()
                for ctype in self.generators.keys()
            }
        
        self.current_session = []
        return self._generate_initial_content(user_id, content_type)
    
    def _generate_initial_content(self, user_id: str, content_type: str) -> List[ContentItem]:
        """
        Generate diverse initial content for cold start
        """
        initial_content = []
        
        # Diverse sampling strategy for exploration
        for i in range(10):
            if content_type == 'mixed':
                ctype = np.random.choice(list(self.generators.keys()))
            else:
                ctype = content_type
            
            # Diverse latent sampling
            if i < 3:
                # Totally random for diversity
                z = torch.randn(1, self.latent_dim)
            elif i < 6:
                # Slightly biased towards common preferences
                z = torch.randn(1, self.latent_dim) * 0.7
            else:
                # Near center of latent space
                z = torch.randn(1, self.latent_dim) * 0.3
            
            content = self.generators[ctype](z)
            
            item = ContentItem(
                content_type=ctype,
                latent=z,
                content=content
            )
            
            initial_content.append(item)
        
        return initial_content
    
    def swipe(self, user_id: str, content_item: ContentItem, 
              action: str, view_duration: float) -> ContentItem:
        """
        Process a swipe and get next content
        Actions: 'like', 'skip', 'share', 'save', 'watch_again'
        """
        # Record engagement
        content_item.user_action = action
        content_item.view_duration = view_duration
        content_item.engagement_score = self._calculate_engagement_score(
            action, view_duration
        )
        
        # Update preference model with this interaction
        self._update_preferences(user_id, content_item)
        
        # Add to history
        self.engagement_history.append(content_item)
        self.current_session.append(content_item)
        
        # Generate next content based on updated preferences
        next_content = self._generate_next_content(user_id)
        
        return next_content
    
    def _generate_next_content(self, user_id: str) -> ContentItem:
        """
        Generate next piece of content based on learned preferences
        Balances exploitation with exploration
        """
        # Decide content type based on recent engagement
        content_type = self._select_content_type(user_id)
        
        # Exploration vs exploitation
        if np.random.random() < 0.2:  # 20% exploration
            # Random content for discovery
            z = torch.randn(1, self.latent_dim)
        else:
            # Generate based on preferences
            z = self._optimize_for_user(user_id, content_type)
        
        # Generate content
        content = self.generators[content_type](z)
        
        return ContentItem(
            content_type=content_type,
            latent=z,
            content=content
        )
    
    def _optimize_for_user(self, user_id: str, content_type: str) -> torch.Tensor:
        """
        Find optimal latent vector for user preferences
        """
        model = self.preference_models[user_id][content_type]
        
        # Start from random point
        z = torch.randn(1, self.latent_dim, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=0.02)
        
        # Quick optimization (needs to be fast for real-time)
        for _ in range(50):
            optimizer.zero_grad()
            score = model(z)
            loss = -score  # Maximize preference
            loss.backward()
            optimizer.step()
            
            # Keep in reasonable bounds
            with torch.no_grad():
                z.clamp_(-3, 3)
        
        return z.detach()
    
    def generate_binge_session(self, user_id: str, duration_minutes: int = 30):
        """
        Generate a binge-worthy session of content
        Maintains engagement with variety and pacing
        """
        print(f"\n📱 Generating {duration_minutes}-minute binge session...")
        
        session_content = []
        elapsed_time = 0
        
        while elapsed_time < duration_minutes * 60:
            # Vary content types for engagement
            if len(session_content) % 5 == 0:
                # Every 5th item is a different type for variety
                content_type = np.random.choice(list(self.generators.keys()))
            else:
                # Otherwise use preferred type
                content_type = self._select_content_type(user_id)
            
            # Generate high-preference content
            z = self._optimize_for_user(user_id, content_type)
            
            # Add some variety
            if len(session_content) % 3 == 0:
                z += torch.randn_like(z) * 0.1
            
            content = self.generators[content_type](z)
            
            item = ContentItem(
                content_type=content_type,
                latent=z,
                content=content
            )
            
            session_content.append(item)
            
            # Simulate viewing time
            if content_type == 'video':
                elapsed_time += np.random.randint(15, 60)
            elif content_type == 'music':
                elapsed_time += np.random.randint(180, 240)
            else:
                elapsed_time += np.random.randint(5, 30)
        
        print(f"Generated {len(session_content)} pieces of content!")
        return session_content
    
    def get_user_insights(self, user_id: str) -> Dict:
        """
        Analyze user preferences without explicit input
        """
        if user_id not in self.preference_models:
            return {"error": "User not found"}
        
        user_history = [item for item in self.engagement_history 
                       if hasattr(item, 'user_id') and item.user_id == user_id]
        
        insights = {
            'preferred_content_types': self._analyze_content_preferences(user_history),
            'engagement_pattern': self._analyze_engagement_pattern(user_history),
            'session_stats': {
                'avg_session_length': np.mean([len(s) for s in self._get_sessions(user_history)]),
                'total_content_consumed': len(user_history)
            },
            'preference_evolution': self._track_preference_evolution(user_id)
        }
        
        return insights
    
    def _calculate_engagement_score(self, action: str, duration: float) -> float:
        """Calculate engagement score from user action"""
        action_scores = {
            'like': 1.0,
            'share': 1.2,
            'save': 1.1,
            'watch_again': 1.3,
            'skip': 0.1
        }
        
        base_score = action_scores.get(action, 0.5)
        
        # Adjust for view duration (normalized)
        duration_factor = min(duration / 30.0, 1.0)  # 30 seconds is "full view"
        
        return base_score * (0.5 + 0.5 * duration_factor)
    
    def _update_preferences(self, user_id: str, content_item: ContentItem):
        """Update preference model based on engagement"""
        model = self.preference_models[user_id][content_item.content_type]
        
        # Simple gradient update based on engagement
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        optimizer.zero_grad()
        predicted_score = model(content_item.latent)
        target_score = torch.tensor([[content_item.engagement_score]])
        
        loss = nn.MSELoss()(predicted_score, target_score)
        loss.backward()
        optimizer.step()
    
    def _select_content_type(self, user_id: str) -> str:
        """Select next content type based on recent engagement"""
        recent = list(self.current_session[-10:])
        if not recent:
            return np.random.choice(list(self.generators.keys()))
        
        # Analyze recent engagement
        type_scores = {}
        for ctype in self.generators.keys():
            type_items = [item for item in recent if item.content_type == ctype]
            if type_items:
                avg_score = np.mean([item.engagement_score for item in type_items])
                type_scores[ctype] = avg_score
            else:
                type_scores[ctype] = 0.5
        
        # Weighted random selection
        types = list(type_scores.keys())
        weights = list(type_scores.values())
        weights = np.array(weights) / sum(weights)
        
        return np.random.choice(types, p=weights)
    
    def _create_preference_model(self):
        """Create a preference model for content type"""
        return nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def _analyze_content_preferences(self, history):
        """Analyze content type preferences"""
        if not history:
            return {}
        
        type_counts = {}
        type_engagement = {}
        
        for item in history:
            ctype = item.content_type
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
            if ctype not in type_engagement:
                type_engagement[ctype] = []
            type_engagement[ctype].append(item.engagement_score)
        
        preferences = {}
        for ctype in type_counts:
            preferences[ctype] = {
                'count': type_counts[ctype],
                'avg_engagement': np.mean(type_engagement[ctype])
            }
        
        return preferences
    
    def _analyze_engagement_pattern(self, history):
        """Analyze user engagement patterns"""
        if not history:
            return {}
        
        actions = [item.user_action for item in history if item.user_action]
        durations = [item.view_duration for item in history]
        
        return {
            'like_rate': actions.count('like') / len(actions) if actions else 0,
            'skip_rate': actions.count('skip') / len(actions) if actions else 0,
            'avg_view_duration': np.mean(durations) if durations else 0,
            'engagement_trend': 'increasing' if len(history) > 10 else 'building'
        }
    
    def _get_sessions(self, history):
        """Split history into sessions"""
        # Simple session detection (gap > 30 minutes)
        sessions = []
        current_session = []
        
        for i, item in enumerate(history):
            current_session.append(item)
            # Session boundary detection would go here
            if i % 20 == 19:  # Simple: every 20 items is a session
                sessions.append(current_session)
                current_session = []
        
        if current_session:
            sessions.append(current_session)
        
        return sessions
    
    def _track_preference_evolution(self, user_id: str):
        """Track how preferences evolve over time"""
        # Simplified version - in production would track actual changes
        return {
            'stability': 'converging',
            'exploration_rate': 0.2,
            'preference_shifts': []
        }


# Example Usage
if __name__ == "__main__":
    print("=== Zero-Prompt Social Media Feed Demo ===")
    
    # Mock generators for different content types
    class MockGenerator:
        def __init__(self, content_type):
            self.content_type = content_type
        
        def __call__(self, z):
            return f"{self.content_type}_content_{hash(z.sum().item()) % 1000}"
    
    # Initialize system
    generators = {
        'video': MockGenerator('video'),
        'music': MockGenerator('music'),
        'image': MockGenerator('image'),
        'story': MockGenerator('story')
    }
    
    feed = ZeroPromptFeed(generators)
    
    # Simulate user session
    user_id = "user_123"
    initial_content = feed.start_new_session(user_id)
    
    print("Initial content served (no prompts!):")
    for i, item in enumerate(initial_content[:5]):
        print(f"{i+1}. {item.content_type}: {item.content}")
    
    # Simulate swiping
    print("\n📱 Simulating user swipes...")
    current_item = initial_content[0]
    
    swipe_actions = ['like', 'skip', 'like', 'share', 'skip', 'like', 'save']
    
    for i, action in enumerate(swipe_actions):
        duration = np.random.randint(5, 30)
        print(f"\nSwipe {i+1}: {action} (viewed for {duration}s)")
        
        next_item = feed.swipe(user_id, current_item, action, duration)
        print(f"Next: {next_item.content_type} - {next_item.content}")
        
        current_item = next_item
    
    # Generate binge session
    print("\n🎬 Generating 30-minute binge session...")
    binge_content = feed.generate_binge_session(user_id, duration_minutes=30)
    
    print(f"\nBinge session preview:")
    for item in binge_content[:10]:
        print(f"- {item.content_type}: {item.content}")
    
    # Show insights
    print("\n📊 User insights (learned without any explicit input!):")
    insights = feed.get_user_insights(user_id)
    for key, value in insights.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    print("\n✨ No prompts needed - just swipe and enjoy!")
    print("The AI learns your preferences and generates infinite personalized content.")