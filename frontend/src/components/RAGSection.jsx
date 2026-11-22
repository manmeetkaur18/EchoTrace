import React, { useState } from 'react';
import { askRAGQuestion } from '../api';

const EXAMPLE_QUESTIONS = [
  'What were the main topics discussed?',
  'What action items or decisions were made?',
  'What was mentioned about the budget or metrics?',
  'Summarize the key points in this recording.',
];

export default function RAGSection({ activeRecordingId, onSeek }) {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [ragResult, setRagResult] = useState(null);

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const res = await askRAGQuestion(query.trim(), activeRecordingId, 5);
      setRagResult(res);
    } catch (err) {
      setError(err.message || 'Failed to generate evidence-grounded answer.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleExampleClick = (example) => {
    setQuery(example);
  };

  return (
    <div className="glass-panel" style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '16px',
      padding: '22px',
    }}>
      {/* Title */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '8px',
            background: 'rgba(99, 102, 241, 0.2)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--accent-primary)',
          }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
          </div>
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: '700' }}>AI Evidence Search & Q&A</h3>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Ask questions grounded strictly in transcript citations
            </p>
          </div>
        </div>

        {activeRecordingId && (
          <span className="badge badge-evidence mono" style={{ fontSize: '0.7rem' }}>
            Scoped to Current Recording
          </span>
        )}
      </div>

      {/* Question Form */}
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '10px' }}>
        <div style={{ flex: 1, position: 'relative' }}>
          <input
            type="text"
            className="input-text"
            placeholder="Ask anything about what was spoken in this recording..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isLoading}
            style={{ paddingRight: '36px' }}
          />
          {query && (
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => setQuery('')}
              style={{
                position: 'absolute',
                right: '6px',
                top: '50%',
                transform: 'translateY(-50%)',
                padding: '4px 6px',
                fontSize: '0.75rem',
              }}
            >
              ✕
            </button>
          )}
        </div>

        <button
          type="submit"
          className="btn btn-primary"
          disabled={isLoading || !query.trim()}
          style={{ minWidth: '110px' }}
        >
          {isLoading ? (
            <>
              <svg className="animate-spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
                <path d="M12 2a10 10 0 0 1 10 10" />
              </svg>
              Searching...
            </>
          ) : (
            <>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="9 18 15 12 9 6" />
              </svg>
              Ask
            </>
          )}
        </button>
      </form>

      {/* Suggested Quick Questions */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Suggestions:</span>
        {EXAMPLE_QUESTIONS.map((ex, idx) => (
          <button
            key={idx}
            type="button"
            className="btn btn-ghost"
            onClick={() => handleExampleClick(ex)}
            style={{
              padding: '3px 10px',
              fontSize: '0.75rem',
              background: 'rgba(255, 255, 255, 0.04)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-full)',
            }}
          >
            {ex}
          </button>
        ))}
      </div>

      {/* Error Message */}
      {error && (
        <div style={{
          padding: '12px 16px',
          borderRadius: 'var(--radius-md)',
          background: 'rgba(244, 63, 94, 0.15)',
          border: '1px solid rgba(244, 63, 94, 0.3)',
          color: '#fda4af',
          fontSize: '0.85rem',
        }}>
          {error}
        </div>
      )}

      {/* Answer & Evidence Box */}
      {ragResult && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
          marginTop: '6px',
          animation: 'fadeIn 0.25s ease',
        }}>
          {/* Answer Box */}
          <div style={{
            padding: '18px 20px',
            borderRadius: 'var(--radius-lg)',
            background: 'rgba(15, 23, 42, 0.9)',
            border: ragResult.has_evidence ? '1px solid rgba(99, 102, 241, 0.35)' : '1px solid rgba(245, 158, 11, 0.3)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
                Grounded Response
              </span>

              {ragResult.has_evidence ? (
                <span className="badge badge-ready">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  Evidence Verified
                </span>
              ) : (
                <span className="badge badge-processing">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="8" x2="12" y2="12" />
                    <line x1="12" y1="16" x2="12.01" y2="16" />
                  </svg>
                  Abstained / Insufficient Evidence
                </span>
              )}
            </div>

            <p style={{
              fontSize: '0.95rem',
              color: '#f8fafc',
              lineHeight: 1.65,
              whiteSpace: 'pre-wrap',
            }}>
              {ragResult.answer}
            </p>
          </div>

          {/* Evidence Citations */}
          {ragResult.sources && ragResult.sources.length > 0 && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                <h4 style={{ fontSize: '0.85rem', fontWeight: '600', color: 'var(--text-secondary)' }}>
                  Supporting Evidence Sources ({ragResult.sources.length}):
                </h4>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Click timestamp to seek video/audio player
                </span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {ragResult.sources.map((source, idx) => (
                  <div
                    key={source.chunk_id || idx}
                    style={{
                      padding: '12px 16px',
                      borderRadius: 'var(--radius-md)',
                      background: 'rgba(30, 41, 59, 0.45)',
                      border: '1px solid var(--border-subtle)',
                      transition: 'border-color 0.15s ease',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '0.75rem', fontWeight: '700', color: 'var(--accent-primary)' }}>
                          Source {idx + 1}
                        </span>
                        {/* Clickable timestamp pill that seeks player! */}
                        <button
                          className="timestamp-pill"
                          onClick={() => onSeek(source.start)}
                          title={`Click to jump media player to ${source.timestamp_range}`}
                        >
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
                            <polygon points="5 3 19 12 5 21 5 3" />
                          </svg>
                          {source.timestamp_range}
                        </button>
                      </div>

                      <span className="mono" style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        Distance: {typeof source.distance === 'number' ? source.distance.toFixed(4) : '--'}
                      </span>
                    </div>

                    <blockquote style={{
                      fontSize: '0.85rem',
                      color: '#cbd5e1',
                      lineHeight: 1.5,
                      borderLeft: '2px solid var(--accent-primary)',
                      paddingLeft: '10px',
                      fontStyle: 'italic',
                    }}>
                      "{source.text}"
                    </blockquote>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
