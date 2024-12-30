import gym
import slimevolleygym
import numpy as np
import os
import time
from PIL import Image
import pygame
from slimevolleygym.slimevolley import PIXEL_MODE, setPixelObsMode

# Force pygame renderer
setattr(slimevolleygym.slimevolley, 'PIXEL_MODE', False)
print(f"PIXEL_MODE set to: {PIXEL_MODE}")

def test_rendering():
    # Test rgb_array mode first
    print("Testing rgb_array mode with GA vs baseline...")
    env = gym.make('SlimeVolley-v0', render_mode='rgb_array')
    env.reset()
    
    # Load GA agent and baseline policy
    from slimevolleygym.mlp import makeSlimePolicyLite
    from slimevolleygym import BaselinePolicy
    
    ga_policy = makeSlimePolicyLite("zoo/ga_sp/ga.json")
    baseline_policy = BaselinePolicy()
    
    # Capture frames of actual gameplay
    frames = []
    obs = env.reset()
    
    try:
        for _ in range(100):  # Capture 100 frames
            action0 = ga_policy.predict(obs)
            action1 = baseline_policy.predict(obs)
            obs, reward, done, info = env.unwrapped.step(action0, action1)
            frame = env.render(mode='rgb_array')  # Explicitly request rgb_array mode
            if frame is not None and isinstance(frame, np.ndarray):
                # Ensure frame is in correct format (HxWx3 uint8)
                if frame.dtype != np.uint8:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                frames.append(frame)
                if len(frames) == 1:  # Debug first frame
                    print(f"First frame: shape={frame.shape}, dtype={frame.dtype}, range=[{frame.min()}, {frame.max()}]")
            if done:
                obs = env.reset()
        
        # Save frames
        os.makedirs('/tmp/frames', exist_ok=True)
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(f'/tmp/frames/frame_{i:03d}.png')
        print(f"Saved {len(frames)} frames to /tmp/frames/")
    except Exception as e:
        print(f"Error during rgb_array rendering: {e}")
    finally:
        env.close()

    print("\nTesting human mode...")
    try:
        env = gym.make('SlimeVolley-v0', render_mode='human')
        env.reset()
        
        # Load GA agent and baseline policy again
        ga_policy = makeSlimePolicyLite("zoo/ga_sp/ga.json")
        baseline_policy = BaselinePolicy()
        
        # Run some frames in human mode
        obs = env.reset()
        start_time = time.time()
        frames_shown = 0
        
        while time.time() - start_time < 5:  # Run for 5 seconds
            action0 = ga_policy.predict(obs)
            action1 = baseline_policy.predict(obs)
            obs, reward, done, info = env.unwrapped.step(action0, action1)
            env.render()
            frames_shown += 1
            if done:
                obs = env.reset()
            time.sleep(1/50)  # Cap at 50 FPS
        
        print(f"Displayed {frames_shown} frames in human mode")
    except Exception as e:
        print(f"Error during human mode rendering: {e}")
    finally:
        env.close()
        
    print("Visualization test complete - check /tmp/frames/ for gameplay frames")

if __name__ == '__main__':
    test_rendering()
