import React, { useState, useEffect, useRef } from 'react';
import Navbar from './components/Navbar';
import RecordingsSidebar from './components/RecordingsSidebar';
import MediaPlayer from './components/MediaPlayer';
import TranscriptViewer from './components/TranscriptViewer';
import RAGSection from './components/RAGSection';
import UploadModal from './components/UploadModal';
import {
  checkBackendHealth,
  listRecordings,
  getRecordingStatus,
  getRecordingTranscript,
  deleteRecording,
} from './api';

export default function App() {
  const [isBackendOnline, setIsBackendOnline] = useState(false);
  const [recordings, setRecordings] = useState([]);
  const [activeRecordingId, setActiveRecordingId] = useState(null);
  const [activeRecording, setActiveRecording] = useState(null);
  const [transcriptChunks, setTranscriptChunks] = useState([]);
  const [isLoadingTranscript, setIsLoadingTranscript] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [targetSeekTime, setTargetSeekTime] = useState(null);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [deleteError, setDeleteError] = useState(null);

  const playerRef = useRef(null);

  // 1. Initial health check & recordings fetch
  const refreshRecordings = async () => {
    try {
      const isOnline = await checkBackendHealth();
      setIsBackendOnline(isOnline);
      if (isOnline) {
        const recList = await listRecordings();
        setRecordings(recList);
        // If no active recording selected yet, select the first one
        if (!activeRecordingId && recList.length > 0) {
          setActiveRecordingId(recList[0].recording_id);
          setActiveRecording(recList[0]);
        }
      }
    } catch (err) {
      console.error('Error refreshing recordings:', err);
    }
  };

  useEffect(() => {
    refreshRecordings();
    const interval = setInterval(refreshRecordings, 15000);
    return () => clearInterval(interval);
  }, []);

  // 2. Load active recording metadata & transcript when selection changes
  useEffect(() => {
    if (!activeRecordingId) {
      setActiveRecording(null);
      setTranscriptChunks([]);
      return;
    }

    let isMounted = true;

    const loadRecordingDetails = async () => {
      try {
        const statusMeta = await getRecordingStatus(activeRecordingId);
        if (!isMounted) return;
        setActiveRecording(statusMeta);

        if (statusMeta.status === 'ready') {
          setIsLoadingTranscript(true);
          const transcriptData = await getRecordingTranscript(activeRecordingId);
          if (isMounted) {
            setTranscriptChunks(transcriptData.chunks || []);
            setIsLoadingTranscript(false);
          }
        } else {
          setTranscriptChunks([]);
        }
      } catch (err) {
        console.error('Failed to load recording details:', err);
        if (isMounted) setIsLoadingTranscript(false);
      }
    };

    loadRecordingDetails();

    return () => {
      isMounted = false;
    };
  }, [activeRecordingId]);

  // 3. Poll processing status if active recording is still processing
  useEffect(() => {
    if (!activeRecording || activeRecording.status !== 'processing') return;

    const pollInterval = setInterval(async () => {
      try {
        const updatedMeta = await getRecordingStatus(activeRecording.recording_id);
        setActiveRecording(updatedMeta);

        // Update item in recordings list
        setRecordings((prev) =>
          prev.map((r) => (r.recording_id === updatedMeta.recording_id ? updatedMeta : r))
        );

        if (updatedMeta.status === 'ready') {
          clearInterval(pollInterval);
          setIsLoadingTranscript(true);
          const transcriptData = await getRecordingTranscript(updatedMeta.recording_id);
          setTranscriptChunks(transcriptData.chunks || []);
          setIsLoadingTranscript(false);
        } else if (updatedMeta.status === 'failed') {
          clearInterval(pollInterval);
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 2500);

    return () => clearInterval(pollInterval);
  }, [activeRecording]);

  // 4. Handle seek action triggered from any timestamp pill
  const handleSeek = (seconds) => {
    setTargetSeekTime(seconds);
    if (playerRef.current) {
      playerRef.current.seekTo(seconds);
    }
  };

  // 5. Handle successful upload
  const handleUploadSuccess = async (newRecordingId) => {
    await refreshRecordings();
    setActiveRecordingId(newRecordingId);
  };

  // 6. Handle recording deletion
  const handleDeleteRecording = async (recordingId) => {
    setDeleteError(null);
    try {
      await deleteRecording(recordingId);
      if (activeRecordingId === recordingId) {
        setActiveRecordingId(null);
        setActiveRecording(null);
        setTranscriptChunks([]);
        setTargetSeekTime(null);
      }
      await refreshRecordings();
    } catch (err) {
      console.error('Delete recording error:', err);
      setDeleteError(err.message || 'Failed to delete recording.');
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', background: 'var(--bg-primary)' }}>
      {/* Top Navbar */}
      <Navbar
        isBackendOnline={isBackendOnline}
        onOpenUpload={() => setIsUploadOpen(true)}
      />

      {/* Main Workspace Layout */}
      <main style={{
        display: 'flex',
        flex: 1,
        padding: '20px 24px',
        gap: '20px',
        overflow: 'hidden',
      }}>
        {/* Left: Recordings Library Sidebar */}
        <RecordingsSidebar
          recordings={recordings}
          activeRecordingId={activeRecordingId}
          onSelectRecording={(id) => {
            setActiveRecordingId(id);
            setTargetSeekTime(null);
          }}
          onDeleteRecording={handleDeleteRecording}
          onRefresh={refreshRecordings}
        />

        {/* Center / Right Workspace */}
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          gap: '20px',
          overflowY: 'auto',
          minWidth: 0,
        }}>
          {/* Deletion Error Feedback Banner */}
          {deleteError && (
            <div style={{
              padding: '12px 18px',
              borderRadius: 'var(--radius-md)',
              background: 'rgba(244, 63, 94, 0.15)',
              border: '1px solid rgba(244, 63, 94, 0.4)',
              color: '#fda4af',
              fontSize: '0.85rem',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}>
              <span>⚠️ {deleteError}</span>
              <button
                className="btn btn-ghost"
                style={{ padding: '2px 8px', fontSize: '0.8rem' }}
                onClick={() => setDeleteError(null)}
              >
                ✕
              </button>
            </div>
          )}

          {/* Status Alert Banner if processing or failed */}
          {activeRecording && activeRecording.status === 'processing' && (
            <div style={{
              padding: '14px 20px',
              borderRadius: 'var(--radius-md)',
              background: 'rgba(245, 158, 11, 0.12)',
              border: '1px solid rgba(245, 158, 11, 0.35)',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
            }}>
              <svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2.5">
                <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
                <path d="M12 2a10 10 0 0 1 10 10" />
              </svg>
              <div>
                <p style={{ fontWeight: '600', color: '#fbbf24', fontSize: '0.9rem' }}>
                  Processing Recording: {activeRecording.original_filename}
                </p>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  Transcribing speech with Groq Whisper, generating Gemini vector embeddings, and indexing in ChromaDB...
                </p>
              </div>
            </div>
          )}

          {activeRecording && activeRecording.status === 'failed' && (
            <div style={{
              padding: '14px 20px',
              borderRadius: 'var(--radius-md)',
              background: 'rgba(244, 63, 94, 0.12)',
              border: '1px solid rgba(244, 63, 94, 0.35)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '12px',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                <div>
                  <p style={{ fontWeight: '600', color: '#fda4af', fontSize: '0.9rem', marginBottom: '2px' }}>
                    Ingestion Failed for {activeRecording.original_filename}
                  </p>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                    {activeRecording.error_message || 'An error occurred during transcription or storage.'}
                  </p>
                </div>
              </div>
              <button
                className="btn btn-secondary"
                style={{
                  fontSize: '0.78rem',
                  padding: '6px 12px',
                  color: '#f43f5e',
                  borderColor: 'rgba(244, 63, 94, 0.4)',
                  flexShrink: 0,
                }}
                onClick={() => {
                  if (window.confirm(`Delete failed recording "${activeRecording.original_filename}"?`)) {
                    handleDeleteRecording(activeRecording.recording_id);
                  }
                }}
              >
                Delete Recording
              </button>
            </div>
          )}

          {/* Upper Half: Media Player */}
          <MediaPlayer
            ref={playerRef}
            recording={activeRecording}
            onTimeUpdate={(t) => setCurrentTime(t)}
            targetSeekTime={targetSeekTime}
          />

          {/* Lower Half: Split between RAG Q&A and Transcript Viewer */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(380px, 1.2fr) minmax(320px, 1fr)',
            gap: '20px',
            alignItems: 'start',
          }}>
            {/* Left Box: RAG Q&A with Evidence Citations */}
            <RAGSection
              activeRecordingId={activeRecordingId}
              onSeek={handleSeek}
            />

            {/* Right Box: Timestamped Transcript Viewer */}
            <TranscriptViewer
              chunks={transcriptChunks}
              currentTime={currentTime}
              onSeek={handleSeek}
              isLoading={isLoadingTranscript}
            />
          </div>
        </div>
      </main>

      {/* Upload & Ingestion Modal */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUploadSuccess={handleUploadSuccess}
      />
    </div>
  );
}
