import React, { useState, useEffect, useRef } from 'react';

export default function TranscriptViewer({
  chunks,
  currentTime,
  onSeek,
  isLoading,
}) {
  const [searchQuery, setSearchQuery] = useState('');
  const [autoScroll, setAutoScroll] = useState(true);
  const activeChunkRef = useRef(null);

  // Filter chunks by search term
  const filteredChunks = chunks.filter((c) =>
    searchQuery.trim() === '' || c.text.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Auto-scroll to active chunk when playback progresses
  useEffect(() => {
    if (autoScroll && activeChunkRef.current) {
      activeChunkRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      });
    }
  }, [currentTime, autoScroll]);

  return (
    <div className="glass-panel" style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      minHeight: '380px',
      overflow: 'hidden',
    }}>
      {/* Header & Search */}
      <div style={{
        padding: '16px 20px',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--accent-primary)' }}>
              <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z" />
              <path d="M6 6h10" />
              <path d="M6 10h10" />
              <path d="M6 14h6" />
            </svg>
            <h3 style={{ fontSize: '0.95rem', fontWeight: '600' }}>Indexed Transcript</h3>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              ({chunks.length} chunks)
            </span>
          </div>

          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
              style={{ accentColor: 'var(--accent-primary)' }}
            />
            Auto-follow playback
          </label>
        </div>

        {/* Filter Input */}
        <div style={{ position: 'relative' }}>
          <input
            type="text"
            className="input-text"
            placeholder="Filter transcript keywords..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{ paddingLeft: '34px', fontSize: '0.8rem', padding: '7px 12px 7px 32px' }}
          />
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }}>
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
        </div>
      </div>

      {/* Chunks List */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '14px 18px',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px',
      }}>
        {isLoading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
            <svg className="animate-spin" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ margin: '0 auto 8px' }}>
              <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
              <path d="M12 2a10 10 0 0 1 10 10" />
            </svg>
            <p style={{ fontSize: '0.85rem' }}>Loading indexed transcript chunks...</p>
          </div>
        ) : filteredChunks.length === 0 ? (
          <div style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            {searchQuery ? 'No transcript chunks match your keyword.' : 'Transcript will appear here once processing completes.'}
          </div>
        ) : (
          filteredChunks.map((chunk) => {
            const isActive = currentTime >= chunk.start && currentTime <= chunk.end;

            return (
              <div
                key={chunk.chunk_id}
                ref={isActive ? activeChunkRef : null}
                style={{
                  padding: '10px 14px',
                  borderRadius: 'var(--radius-md)',
                  background: isActive ? 'rgba(99, 102, 241, 0.14)' : 'rgba(15, 23, 42, 0.5)',
                  border: isActive ? '1px solid var(--accent-primary)' : '1px solid var(--border-subtle)',
                  transition: 'all 0.15s ease',
                  position: 'relative',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <button
                    className="timestamp-pill"
                    onClick={() => onSeek(chunk.start)}
                    title={`Click to jump player to ${chunk.timestamp_range}`}
                  >
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
                      <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                    {chunk.timestamp_range}
                  </button>

                  {isActive && (
                    <span className="badge badge-evidence animate-pulse" style={{ fontSize: '0.65rem' }}>
                      Playing
                    </span>
                  )}
                </div>

                <p
                  onClick={() => onSeek(chunk.start)}
                  style={{
                    fontSize: '0.85rem',
                    color: isActive ? '#f8fafc' : '#cbd5e1',
                    lineHeight: 1.55,
                    cursor: 'pointer',
                  }}
                  title="Click to seek player here"
                >
                  {chunk.text}
                </p>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
