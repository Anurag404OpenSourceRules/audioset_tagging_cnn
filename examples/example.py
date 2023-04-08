import librosa
# import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels

audio_path = 'R9_ZSCveAHg_7s.wav'
(audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
audio = audio[None, :]  # (batch_size, segment_samples)

print('------ Audio tagging ------')
at = AudioTagging(checkpoint_path=None, device='cuda')
(clipwise_output, embedding) = at.inference(audio)

print('------ Sound event detection ------')
sed = SoundEventDetection(checkpoint_path=None, device='cuda')
framewise_output = sed.inference(audio)
