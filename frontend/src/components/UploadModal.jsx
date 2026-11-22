import React, { useState, useRef } from 'react';
import { uploadMediaFile, ingestMediaUrl } from '../api';

export function isRestrictedPlatformUrl(urlString) {
  if (!urlString || typeof urlString !== 'string') return false;
  const trimmed = urlString.trim();
  if (!trimmed) return false;
  try {
    const normalized = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
    const parsed = new URL(normalized);
    const host = parsed.hostname.toLowerCase();

    // Recognize legitimate YouTube hostnames (e.g. youtube.com, www.youtube.com, m.youtube.com, youtu.be)
    const isYouTube = host === 'youtube.com' || host.endsWith('.youtube.com') || host === 'youtu.be' || host.endsWith('.youtu.be');

    // Recognize legitimate TED hostnames (e.g. ted.com, www.ted.com)
    const isTed = host === 'ted.com' || host.endsWith('.ted.com');

    return isYouTube || isTed;
  } catch (_) {
    return false;
  }
}

export default function UploadModal({ isOpen, onClose, onUploadSuccess }) {
  const [activeTab, setActiveTab] = useState('file'); // 'file' | 'url'
  const [file, setFile] = useState(null);
  const [url, setUrl] = useState('');
  const [episodeFilter, setEpisodeFilter] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const isYouTubeOrTed = isRestrictedPlatformUrl(url);

  if (!isOpen) return null;

  const handleFileDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selected = e.dataTransfer.files[0];
      setFile(selected);
      setError(null);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setIsSubmitting(true);

    try {
      if (activeTab === 'file') {
        if (!file) {
          throw new Error('Please select an audio or video file to upload.');
        }
        const res = await uploadMediaFile(file);
        onUploadSuccess(res.recording_id);
        onClose();
      } else {
        if (!url.trim()) {
          throw new Error('Please enter a media URL or podcast RSS feed URL.');
        }
        if (isYouTubeOrTed) {
          throw new Error('Direct YouTube and TED downloading is restricted by platform policy. Please save the media file locally and upload it using the "Upload File" tab.');
        }
        const res = await ingestMediaUrl(url.trim(), episodeFilter.trim() || null);
        onUploadSuccess(res.recording_id);
        onClose();
      }
    } catch (err) {
      setError(err.message || 'Operation failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0, 0, 0, 0.75)',
      backdropFilter: 'blur(6px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 100,
      padding: '20px',
    }}>
      <div className="glass-panel" style={{
        width: '100%',
        maxWidth: '520px',
        padding: '28px',
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-active)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h2 style={{ fontSize: '1.2rem', fontWeight: '700' }}>Ingest Recording</h2>
          <button
            className="btn btn-ghost"
            style={{ padding: '4px 8px' }}
            onClick={onClose}
            disabled={isSubmitting}
          >
            ✕
          </button>
        </div>

        {/* Tab Selector */}
        <div style={{
          display: 'flex',
          gap: '8px',
          background: 'var(--bg-primary)',
          padding: '4px',
          borderRadius: 'var(--radius-md)',
          marginBottom: '20px',
        }}>
          <button
            type="button"
            className="btn"
            style={{
              flex: 1,
              background: activeTab === 'file' ? 'var(--bg-tertiary)' : 'transparent',
              color: activeTab === 'file' ? 'var(--text-primary)' : 'var(--text-secondary)',
              fontWeight: activeTab === 'file' ? '600' : '400',
            }}
            onClick={() => { setActiveTab('file'); setError(null); }}
          >
            Upload File
          </button>
          <button
            type="button"
            className="btn"
            style={{
              flex: 1,
              background: activeTab === 'url' ? 'var(--bg-tertiary)' : 'transparent',
              color: activeTab === 'url' ? 'var(--text-primary)' : 'var(--text-secondary)',
              fontWeight: activeTab === 'url' ? '600' : '400',
            }}
            onClick={() => { setActiveTab('url'); setError(null); }}
          >
            Media URL / RSS
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          {activeTab === 'file' ? (
            <div>
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleFileDrop}
                onClick={() => fileInputRef.current?.click()}
                style={{
                  border: '2px dashed var(--border-active)',
                  borderRadius: 'var(--radius-lg)',
                  padding: '36px 20px',
                  textAlign: 'center',
                  background: 'rgba(99, 102, 241, 0.04)',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  marginBottom: '16px',
                }}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".mp4,.mp3,.wav,.m4a,video/mp4,audio/mpeg,audio/wav,audio/x-m4a"
                  style={{ display: 'none' }}
                  onChange={handleFileSelect}
                />
                <div style={{
                  width: '48px',
                  height: '48px',
                  borderRadius: '12px',
                  background: 'rgba(99, 102, 241, 0.15)',
                  color: 'var(--accent-primary)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 12px',
                }}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
                    <path d="M12 12v9" />
                    <path d="m16 16-4-4-4 4" />
                  </svg>
                </div>
                {file ? (
                  <div>
                    <p style={{ fontWeight: '600', color: '#a5b4fc', marginBottom: '4px' }}>
                      {file.name}
                    </p>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                ) : (
                  <div>
                    <p style={{ fontWeight: '500', marginBottom: '6px' }}>
                      Click to choose or drag & drop media file
                    </p>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                      Supported formats: MP4, MP3, WAV, M4A (Max 100 MB)
                    </p>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginBottom: '20px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Direct Media URL or Podcast RSS Feed
                </label>
                <input
                  type="url"
                  className="input-text"
                  placeholder="https://example.com/audio.mp3 or https://podcast.com/feed.xml"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  disabled={isSubmitting}
                />
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Accepts direct media files (.mp3, .wav, .m4a, .mp4) or podcast RSS XML feeds.
                </p>
              </div>

              {isYouTubeOrTed && (
                <div style={{
                  padding: '10px 14px',
                  borderRadius: 'var(--radius-md)',
                  background: 'rgba(245, 158, 11, 0.12)',
                  border: '1px solid rgba(245, 158, 11, 0.35)',
                  color: '#fbbf24',
                  fontSize: '0.8rem',
                  lineHeight: '1.4',
                }}>
                  <strong>Notice:</strong> YouTube and TED URLs do not provide direct media files and cannot be downloaded directly due to platform streaming protections. To index this recording, save the audio/video file locally and upload it using the <strong>Upload File</strong> tab.
                </div>
              )}
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Episode Title Filter (Optional for podcasts)
                </label>
                <input
                  type="text"
                  className="input-text"
                  placeholder="e.g. Episode 42 (defaults to latest episode)"
                  value={episodeFilter}
                  onChange={(e) => setEpisodeFilter(e.target.value)}
                  disabled={isSubmitting}
                />
              </div>
            </div>
          )}

          {error && (
            <div style={{
              padding: '10px 14px',
              borderRadius: 'var(--radius-md)',
              background: 'rgba(244, 63, 94, 0.15)',
              border: '1px solid rgba(244, 63, 94, 0.3)',
              color: '#fda4af',
              fontSize: '0.85rem',
              marginBottom: '16px',
            }}>
              {error}
            </div>
          )}

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onClose}
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={isSubmitting || (activeTab === 'file' && !file) || (activeTab === 'url' && !url.trim())}
            >
              {isSubmitting ? (
                <>
                  <svg className="animate-spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
                    <path d="M12 2a10 10 0 0 1 10 10" />
                  </svg>
                  Uploading & Starting Ingestion...
                </>
              ) : (
                'Start Ingestion'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
