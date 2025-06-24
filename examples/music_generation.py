"""
PLGL Music Generation Example

This example shows how to apply PLGL to music generation using a VAE-based
music model. Users rate generated melodies, and PLGL learns their musical
preferences to generate personalized compositions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import pretty_midi
from dataclasses import dataclass


@dataclass
class MusicSample:
    """Container for music samples with metadata"""
    latent: torch.Tensor
    notes: torch.Tensor  # Piano roll representation
    rating: float
    tempo: int = 120
    key: str = "C"
    
    def to_midi(self) -> pretty_midi.PrettyMIDI:
        """Convert piano roll to MIDI"""
        midi = pretty_midi.PrettyMIDI(initial_tempo=self.tempo)
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        # Convert piano roll to notes
        piano_roll = self.notes.cpu().numpy()
        for time_step in range(piano_roll.shape[1]):
            for pitch in range(piano_roll.shape[0]):
                if piano_roll[pitch, time_step] > 0.5:
                    # Create note
                    start_time = time_step * 0.125  # 8th note resolution
                    end_time = start_time + 0.125
                    note = pretty_midi.Note(
                        velocity=int(piano_roll[pitch, time_step] * 127),
                        pitch=pitch + 21,  # Start from A0
                        start=start_time,
                        end=end_time
                    )
                    instrument.notes.append(note)
        
        midi.instruments.append(instrument)
        return midi


class MusicVAE(nn.Module):
    """
    Simplified Music VAE for demonstration
    Generates 4-bar melodies with 128 pitches x 32 time steps
    """
    
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(12, 4), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(12, 4), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 32 * 8, 512),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 32 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 32, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=(12, 4), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(12, 4), stride=(2, 2), padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class MusicPreferenceLearner:
    """
    PLGL implementation for music generation
    Learns user's musical preferences and generates personalized melodies
    """
    
    def __init__(self, music_vae: MusicVAE, device: str = 'cuda'):
        self.vae = music_vae.to(device)
        self.device = device
        self.latent_dim = music_vae.latent_dim
        
        # Preference model for music
        self.preference_model = nn.Sequential(
            nn.Linear(128 * 32, 512),  # Flattened piano roll
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)
        
        self.samples = []
        self.optimizer = torch.optim.Adam(self.preference_model.parameters(), lr=0.001)
    
    def generate_melody(self, z: torch.Tensor = None) -> torch.Tensor:
        """Generate melody from latent code"""
        if z is None:
            z = torch.randn(1, self.latent_dim, device=self.device)
        
        with torch.no_grad():
            melody = self.vae.decode(z)
        
        return melody.squeeze(0)
    
    def extract_musical_features(self, melody: torch.Tensor) -> Dict[str, float]:
        """Extract interpretable musical features"""
        piano_roll = melody.squeeze().cpu().numpy()
        
        features = {
            'pitch_range': self._compute_pitch_range(piano_roll),
            'note_density': self._compute_note_density(piano_roll),
            'rhythmic_complexity': self._compute_rhythmic_complexity(piano_roll),
            'melodic_contour': self._compute_melodic_contour(piano_roll),
            'harmonic_consistency': self._compute_harmonic_consistency(piano_roll)
        }
        
        return features
    
    def collect_preferences_interactive(self, n_samples: int = 30) -> List[MusicSample]:
        """
        Collect preferences with musical diversity
        Ensures samples cover different musical characteristics
        """
        samples = []
        
        # Generate diverse musical samples
        for i in range(n_samples):
            # Vary generation to ensure musical diversity
            if i % 3 == 0:
                # High pitch, sparse notes
                z = torch.randn(1, self.latent_dim, device=self.device)
                z[:, :50] *= 2.0  # Emphasize certain latent dimensions
            elif i % 3 == 1:
                # Low pitch, dense notes
                z = torch.randn(1, self.latent_dim, device=self.device)
                z[:, 50:100] *= 2.0
            else:
                # Random
                z = torch.randn(1, self.latent_dim, device=self.device)
            
            melody = self.generate_melody(z)
            
            # Simulate user rating (would be actual UI)
            features = self.extract_musical_features(melody)
            rating = self._simulate_music_preference(features)
            
            sample = MusicSample(
                latent=z,
                notes=melody,
                rating=rating
            )
            
            samples.append(sample)
            self.samples.append(sample)
            
            print(f"Sample {i+1}/{n_samples} - Rating: {rating:.2f}")
        
        return samples
    
    def train_preference_model(self, epochs: int = 100):
        """Train the musical preference model"""
        if len(self.samples) < 10:
            raise ValueError("Need at least 10 samples to train")
        
        # Prepare data
        X = torch.stack([s.notes.flatten() for s in self.samples])
        y = torch.tensor([s.rating for s in self.samples], device=self.device)
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.preference_model(X).squeeze()
            loss = nn.BCELoss()(predictions, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    def generate_personalized_melody(
        self, 
        optimization_steps: int = 500,
        temperature: float = 1.0
    ) -> MusicSample:
        """Generate a melody optimized for user preferences"""
        
        # Start from random latent code
        z = torch.randn(1, self.latent_dim, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=0.01)
        
        best_z = None
        best_score = -1
        
        for step in range(optimization_steps):
            optimizer.zero_grad()
            
            # Generate melody
            melody = self.vae.decode(z)
            
            # Score with preference model
            score = self.preference_model(melody.flatten())
            
            # Optimize (maximize score)
            loss = -score
            loss.backward()
            optimizer.step()
            
            # Track best
            if score.item() > best_score:
                best_score = score.item()
                best_z = z.clone().detach()
            
            # Add noise for exploration
            with torch.no_grad():
                z += torch.randn_like(z) * 0.01 * temperature
        
        # Generate final melody
        final_melody = self.generate_melody(best_z)
        
        return MusicSample(
            latent=best_z,
            notes=final_melody,
            rating=best_score
        )
    
    def generate_playlist(
        self, 
        n_songs: int = 10,
        diversity: float = 0.5
    ) -> List[MusicSample]:
        """Generate a personalized playlist with controlled diversity"""
        
        playlist = []
        
        for i in range(n_songs):
            # Alternate between exploitation and exploration
            if i % 2 == 0 or len(playlist) == 0:
                # Generate high-preference melody
                melody = self.generate_personalized_melody()
            else:
                # Generate with diversity constraint
                z = torch.randn(1, self.latent_dim, device=self.device)
                
                # Ensure different from previous
                if playlist:
                    prev_latents = torch.cat([s.latent for s in playlist])
                    distances = torch.norm(z - prev_latents, dim=1)
                    
                    # Resample if too similar
                    while distances.min() < diversity:
                        z = torch.randn(1, self.latent_dim, device=self.device)
                        distances = torch.norm(z - prev_latents, dim=1)
                
                melody_tensor = self.generate_melody(z)
                score = self.preference_model(melody_tensor.flatten()).item()
                
                melody = MusicSample(
                    latent=z,
                    notes=melody_tensor,
                    rating=score
                )
            
            playlist.append(melody)
            print(f"Generated song {i+1}/{n_songs} - Score: {melody.rating:.2f}")
        
        return playlist
    
    def _compute_pitch_range(self, piano_roll: np.ndarray) -> float:
        """Compute the pitch range of the melody"""
        active_pitches = np.where(piano_roll.sum(axis=1) > 0)[0]
        if len(active_pitches) > 0:
            return (active_pitches.max() - active_pitches.min()) / 128.0
        return 0.0
    
    def _compute_note_density(self, piano_roll: np.ndarray) -> float:
        """Compute the density of notes"""
        return (piano_roll > 0.5).sum() / piano_roll.size
    
    def _compute_rhythmic_complexity(self, piano_roll: np.ndarray) -> float:
        """Compute rhythmic complexity based on note onset patterns"""
        onsets = np.diff(piano_roll.sum(axis=0))
        return np.std(onsets)
    
    def _compute_melodic_contour(self, piano_roll: np.ndarray) -> float:
        """Compute melodic contour (up/down movement)"""
        active_notes = []
        for t in range(piano_roll.shape[1]):
            active = np.where(piano_roll[:, t] > 0.5)[0]
            if len(active) > 0:
                active_notes.append(active.mean())
        
        if len(active_notes) > 1:
            contour = np.diff(active_notes)
            return np.std(contour)
        return 0.0
    
    def _compute_harmonic_consistency(self, piano_roll: np.ndarray) -> float:
        """Estimate harmonic consistency"""
        # Simplified: check for common intervals
        intervals = []
        for t in range(piano_roll.shape[1]):
            active = np.where(piano_roll[:, t] > 0.5)[0]
            if len(active) > 1:
                intervals.extend(np.diff(active))
        
        if intervals:
            # Common intervals in music (octave, fifth, third)
            common_intervals = [12, 7, 4, 3]
            consistency = sum(1 for i in intervals if i % 12 in common_intervals)
            return consistency / len(intervals)
        return 0.5
    
    def _simulate_music_preference(self, features: Dict[str, float]) -> float:
        """
        Simulate user preference based on musical features
        In real implementation, this would be actual user ratings
        """
        # Example: User prefers moderate complexity, wide range, harmonic music
        score = 0.5
        
        # Prefer moderate note density (not too sparse or dense)
        score += 0.2 * (1 - abs(features['note_density'] - 0.3))
        
        # Prefer wider pitch range
        score += 0.2 * features['pitch_range']
        
        # Prefer harmonic consistency
        score += 0.3 * features['harmonic_consistency']
        
        # Prefer moderate rhythmic complexity
        score += 0.1 * (1 - abs(features['rhythmic_complexity'] - 0.5))
        
        # Add some randomness
        score += 0.1 * np.random.random()
        
        return np.clip(score, 0, 1)


# Example usage
if __name__ == "__main__":
    # Initialize music VAE
    music_vae = MusicVAE(latent_dim=256)
    
    # Create preference learner
    learner = MusicPreferenceLearner(music_vae)
    
    print("=== Music Preference Learning Demo ===\n")
    
    # Collect preferences
    print("Collecting musical preferences...")
    preferences = learner.collect_preferences_interactive(n_samples=30)
    
    # Train preference model
    print("\nTraining preference model...")
    learner.train_preference_model(epochs=100)
    
    # Generate personalized melody
    print("\nGenerating your personalized melody...")
    personalized = learner.generate_personalized_melody()
    print(f"Generated melody with preference score: {personalized.rating:.3f}")
    
    # Save as MIDI
    midi = personalized.to_midi()
    midi.write('personalized_melody.mid')
    print("Saved as 'personalized_melody.mid'")
    
    # Generate playlist
    print("\nGenerating personalized playlist...")
    playlist = learner.generate_playlist(n_songs=5, diversity=0.4)
    
    # Save playlist
    for i, song in enumerate(playlist):
        midi = song.to_midi()
        midi.write(f'playlist_song_{i+1}.mid')
    
    print(f"\nSaved {len(playlist)} songs to MIDI files!")
    print("\nAverage preference score:", np.mean([s.rating for s in playlist]))