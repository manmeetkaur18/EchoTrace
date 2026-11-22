import React from 'react';

function formatSeconds(secs) {
  if (!secs || isNaN(secs)) return '--:--';
  const total = Math.floor(secs);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

export default function RecordingsSidebar({
  recordings,
  activeRecordingId,
  onSelectRecording,
  onRefresh,
}) {
  return (
    <aside className="glass-panel" style={{
      width: '320px',
      display: 'flex',
      flexDirection: 'column',
      height: 'calc(100vh - 90px)',
      overflow: 'hidden',
    }}>
      {/* Sidebar Header */}
      <div style={{
        padding: '16px 20px',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--accent-primary)' }}>
            <rect width="20" height="15" x="2" y="7" rx="2" ry="2" />
            <polyline points="17 2 12 7 7 2" />
          </svg>
          <h2 style={{ fontSize: '0.95rem', fontWeight: '600' }}>Recordings Library</h2>
        </div>
        <button
          className="btn btn-ghost"
          style={{ padding: '4px', borderRadius: 'var(--radius-sm)' }}
          title="Refresh recordings"
          onClick={onRefresh}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
            <path d="M3 3v5h5" />
            <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
            <path d="M16 21h5v-5" />
          </svg>
        </button>
      </div>

      {/* Recordings List */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '12px',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
      }}>
        {recordings.length === 0 ? (
          <div style={{
            padding: '32px 16px',
            textAlign: 'center',
            color: 'var(--text-muted)',
            fontSize: '0.85rem',
          }}>
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ margin: '0 auto 10px', opacity: 0.4 }}>
              <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
              <polyline points="14 2 14 8 20 8" />
            </svg>
            <p style={{ fontWeight: '500', marginBottom: '4px' }}>No recordings yet</p>
            <p style={{ fontSize: '0.75rem' }}>Upload media to index and ask questions.</p>
          </div>
        ) : (
          recordings.map((rec) => {
            const isActive = rec.recording_id === activeRecordingId;
            const isProcessing = rec.status === 'processing';
            const isReady = rec.status === 'ready';

            return (
              <div
                key={rec.recording_id}
                onClick={() => onSelectRecording(rec.recording_id)}
                style={{
                  padding: '12px 14px',
                  borderRadius: 'var(--radius-md)',
                  background: isActive ? 'rgba(99, 102, 241, 0.16)' : 'var(--bg-secondary)',
                  border: isActive ? '1px solid var(--accent-primary)' : '1px solid var(--border-subtle)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  position: 'relative',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '8px', marginBottom: '6px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0 }}>
                    {rec.media_type === 'video' ? (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a855f7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                        <rect width="18" height="14" x="3" y="5" rx="2" />
                        <path d="m21 15-4-3 4-3v6z" />
                      </svg>
                    ) : (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                        <path d="M9 18V5l12-2v13" />
                        <circle cx="6" cy="18" r="3" />
                        <circle cx="18" cy="16" r="3" />
                      </svg>
                    )}
                    <span style={{
                      fontWeight: isActive ? '600' : '500',
                      fontSize: '0.85rem',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}>
                      {rec.original_filename}
                    </span>
                  </div>

                  {/* Status Badge */}
                  <span className={`badge badge-${rec.status}`} style={{ flexShrink: 0 }}>
                    {isProcessing && (
                      <svg className="animate-spin" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
                        <path d="M12 2a10 10 0 0 1 10 10" />
                      </svg>
                    )}
                    {rec.status}
                  </span>
                </div>

                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  fontSize: '0.75rem',
                  color: 'var(--text-muted)',
                }}>
                  <span className="mono">
                    {formatSeconds(rec.duration_seconds)}
                  </span>
                  <span>
                    {rec.chunk_count > 0 ? `${rec.chunk_count} chunks` : ''}
                  </span>
                </div>

                {rec.status === 'failed' && rec.error_message && (
                  <p style={{
                    fontSize: '0.72rem',
                    color: '#fda4af',
                    marginTop: '6px',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }} title={rec.error_message}>
                    ⚠️ {rec.error_message}
                  </p>
                )}
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
}
