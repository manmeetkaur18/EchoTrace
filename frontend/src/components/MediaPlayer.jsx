import React, { useRef, useEffect, useState, forwardRef, useImperativeHandle } from 'react';
import { getMediaStreamUrl } from '../api';

function formatTime(seconds) {
  if (!seconds || isNaN(seconds)) return '00:00';
  const total = Math.floor(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

const MediaPlayer = forwardRef(function MediaPlayer(
  { recording, onTimeUpdate, targetSeekTime },
  ref
) {
  const mediaRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(recording?.duration_seconds || 0);

  // Expose seekTo method through ref
  useImperativeHandle(ref, () => ({
    seekTo: (timeInSeconds) => {
      if (mediaRef.current && typeof timeInSeconds === 'number') {
        mediaRef.current.currentTime = timeInSeconds;
        mediaRef.current.play().catch(() => {});
        setIsPlaying(true);
      }
    },
    getCurrentTime: () => mediaRef.current?.currentTime || 0,
  }));

  // Handle external seek triggers
  useEffect(() => {
    if (targetSeekTime !== null && targetSeekTime !== undefined && mediaRef.current) {
      mediaRef.current.currentTime = targetSeekTime;
      mediaRef.current.play().catch(() => {});
      setIsPlaying(true);
    }
  }, [targetSeekTime]);

  // Update current time & emit to parent
  const handleTimeUpdate = () => {
    if (mediaRef.current) {
      const cur = mediaRef.current.currentTime;
      setCurrentTime(cur);
      if (onTimeUpdate) {
        onTimeUpdate(cur);
      }
    }
  };

  const handleLoadedMetadata = () => {
    if (mediaRef.current) {
      setDuration(mediaRef.current.duration || recording?.duration_seconds || 0);
    }
  };

  const togglePlay = () => {
    if (!mediaRef.current) return;
    if (isPlaying) {
      mediaRef.current.pause();
      setIsPlaying(false);
    } else {
      mediaRef.current.play().catch(() => {});
      setIsPlaying(true);
    }
  };

  const handleSeekSlider = (e) => {
    const newTime = parseFloat(e.target.value);
    if (mediaRef.current) {
      mediaRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    }
  };

  if (!recording) {
    return (
      <div className="glass-panel" style={{
        padding: '40px 20px',
        textAlign: 'center',
        color: 'var(--text-muted)',
      }}>
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ margin: '0 auto 12px', opacity: 0.4 }}>
          <polygon points="5 3 19 12 5 21 5 3" />
        </svg>
        <p style={{ fontWeight: '500' }}>Select or upload a recording to play</p>
      </div>
    );
  }

  const streamUrl = getMediaStreamUrl(recording.recording_id);
  const isVideo = recording.media_type === 'video';

  return (
    <div className="glass-panel" style={{
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Video / Audio Container */}
      <div style={{
        background: '#000',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        minHeight: isVideo ? '320px' : '90px',
        maxHeight: isVideo ? '460px' : '100px',
      }}>
        {isVideo ? (
          <video
            ref={mediaRef}
            src={streamUrl}
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleLoadedMetadata}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            controls={false}
            style={{
              width: '100%',
              maxHeight: '440px',
              objectFit: 'contain',
              display: 'block',
            }}
          />
        ) : (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '16px',
            padding: '24px',
            width: '100%',
            background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.95))',
          }}>
            <audio
              ref={mediaRef}
              src={streamUrl}
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              controls={false}
            />
            <div style={{
              width: '44px',
              height: '44px',
              borderRadius: '50%',
              background: 'rgba(99, 102, 241, 0.2)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--accent-primary)',
            }}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 18V5l12-2v13" />
                <circle cx="6" cy="18" r="3" />
                <circle cx="18" cy="16" r="3" />
              </svg>
            </div>
            <div style={{ minWidth: 0, flex: 1 }}>
              <p style={{ fontWeight: '600', fontSize: '0.95rem', color: '#f1f5f9', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {recording.original_filename}
              </p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Audio Stream (HTTP Range Enabled)
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Custom Media Controls */}
      <div style={{
        padding: '12px 18px',
        background: 'var(--bg-secondary)',
        borderTop: '1px solid var(--border-subtle)',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
      }}>
        {/* Progress Timeline Slider */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', minWidth: '40px' }}>
            {formatTime(currentTime)}
          </span>
          <input
            type="range"
            min="0"
            max={duration || 100}
            step="0.1"
            value={currentTime}
            onChange={handleSeekSlider}
            style={{
              flex: 1,
              accentColor: 'var(--accent-primary)',
              cursor: 'pointer',
              height: '5px',
            }}
          />
          <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)', minWidth: '40px', textAlign: 'right' }}>
            {formatTime(duration)}
          </span>
        </div>

        {/* Playback Action Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button
              className="btn btn-primary"
              onClick={togglePlay}
              style={{
                width: '36px',
                height: '36px',
                padding: 0,
                borderRadius: '50%',
              }}
              title={isPlaying ? 'Pause' : 'Play'}
            >
              {isPlaying ? (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="4" width="4" height="16" rx="1" />
                  <rect x="14" y="4" width="4" height="16" rx="1" />
                </svg>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginLeft: '2px' }}>
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
              )}
            </button>

            <button
              className="btn btn-ghost"
              onClick={() => {
                if (mediaRef.current) {
                  mediaRef.current.currentTime = Math.max(0, mediaRef.current.currentTime - 5);
                }
              }}
              title="Rewind 5s"
              style={{ padding: '6px' }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M11 17l-5-5 5-5M18 17l-5-5 5-5" />
              </svg>
            </button>

            <button
              className="btn btn-ghost"
              onClick={() => {
                if (mediaRef.current) {
                  mediaRef.current.currentTime = Math.min(duration, mediaRef.current.currentTime + 5);
                }
              }}
              title="Forward 5s"
              style={{ padding: '6px' }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M13 17l5-5-5-5M6 17l5-5-5-5" />
              </svg>
            </button>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {isVideo ? 'MP4 Video' : 'Audio Stream'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
});

export default MediaPlayer;
