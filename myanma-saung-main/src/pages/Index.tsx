import { useEffect, useRef, useState } from 'react';
import AnimatedBackground from '@/components/AnimatedBackground';
import SiteNav from '@/components/SiteNav';
import VideoUploadZone from '@/components/VideoUploadZone';
import SoundWaveLoader from '@/components/SoundWaveLoader';
import MusicalNotesDisplay from '@/components/MusicalNotesDisplay';
import { CheckCircle2, Gauge, Settings2, UploadCloud, Zap } from 'lucide-react';

type ProcessingStatus = 'idle' | 'analyzing' | 'detecting' | 'generating' | 'complete';
type ProcessingMode = 'fast' | 'accurate';

interface PredictionResponse {
  predicted_video_url?: string | null;
  ksy_notes?: string[];
  frames_processed?: number;
  final_codec?: string;
  input_has_audio_track?: boolean;
  has_audio_track?: boolean;
  touch_events?: TouchMappingPayload[];
  strike_results?: StrikeResultPayload[];
  right_decision_events_json_path?: string | null;
  right_strike_events_json_path?: string | null;
  right_audio_decision_events_json_path?: string | null;
  right_audio_strike_events_json_path?: string | null;
  right_av_decision_events_json_path?: string | null;
  right_av_strike_events_json_path?: string | null;
  right_audio_decision_events?: AudioDecisionEventPayload[];
  right_audio_strike_events?: Record<string, unknown>[];
  right_av_decision_events?: Record<string, unknown>[];
  right_av_strike_events?: AvStrikeEventPayload[];
  right_av_alternating_on_off_slots_json_path?: string | null;
  right_av_alternating_on_off_slots?: AlternatingOnOffSlotsPayload | null;
  right_audio_decision_events_count?: number;
  right_audio_strike_events_count?: number;
  right_av_decision_events_count?: number;
  right_av_strike_events_count?: number;
  av_inference?: {
    error?: string;
    audio_error?: string;
    audio_enabled?: boolean;
    fusion_mode?: string;
    audio_decision_mode?: string;
    right_audio_decision_events_count?: number;
    right_audio_strike_events_count?: number;
    right_av_decision_events_count?: number;
    right_av_strike_events_count?: number;
  } | null;
  analysis_debug_report_path?: string | null;
  analysis_debug_report_url?: string | null;
  analysis_debug_snapshot_count?: number;
  analysis_debug_summary?: {
    flag_counts?: Record<string, number>;
    audio_status_counts?: Record<string, number>;
    fusion_status_counts?: Record<string, number>;
  } | null;
  run_profile?: {
    fast_mode?: boolean;
    audio_decision_mode?: string | null;
  } | null;
}

interface PredictionJobResponse {
  job_id?: string;
  status?: 'queued' | 'processing' | 'complete' | 'failed' | string;
  stage?: string;
  file_name?: string;
  result?: PredictionResponse | null;
  error?: string | null;
}

interface StrikeMetricsPayload {
  candidate_score?: number;
  peak?: number;
  duration?: number;
  impulse?: number;
  vibrates?: boolean;
}

interface StrikeDecisionDebugPayload {
  decision_reason?: string;
}

interface StrikeResultPayload {
  event_time?: number;
  finger_type?: string;
  touched_id?: number;
  struck_id?: number | null;
  label?: string;
  best_metrics?: StrikeMetricsPayload;
  decision_debug?: StrikeDecisionDebugPayload;
}

interface TouchMappingPayload {
  time_sec?: number;
  timestamp_sec?: number;
  frame_index?: number;
  hand_side?: string;
  hand?: string;
  finger_type?: string;
  touched_string_id?: number;
  string_id?: number;
  touch_conf?: number;
  distance_px?: number;
  contact_x?: number;
  contact_y?: number;
}

interface AudioDecisionTouchPayload {
  timestamp_sec?: number | null;
  frame_index?: number | null;
  hand_side?: string;
  finger_type?: string;
  touched_string_id?: number | null;
  touch_conf?: number | null;
  contact_x?: number | null;
  contact_y?: number | null;
  finger_x?: number | null;
  finger_y?: number | null;
  distance_px?: number | null;
}

interface AudioDecisionAudioPayload {
  status?: string;
  touch_time_sec?: number | null;
  onset_time_sec?: number | null;
  onset_score?: number | null;
  pitch_backend?: string | null;
  f0_hz?: number | null;
  pitch_conf?: number | null;
  matched_string_id?: number | null;
  cents_error?: number | null;
}

interface AudioDecisionDecisionPayload {
  struck_string_id?: number | null;
  confidence?: number;
  confidence_label?: string;
  reject_reason?: string | null;
}

interface AudioDecisionEventPayload {
  event_id?: string;
  touch?: AudioDecisionTouchPayload;
  audio?: AudioDecisionAudioPayload;
  decision?: AudioDecisionDecisionPayload;
}

interface AvStrikeEventPayload {
  event_id?: string;
  timestamp_sec?: number;
  frame_index?: number | null;
  finger_type?: string;
  touched_string_id?: number | null;
  struck_string_id?: number | null;
  peak_frame?: number | null;
  confidence?: number;
  confidence_label?: string;
  strategy?: string;
}

interface TimelineEventPayload {
  source: 'av' | 'vibration';
  event_time: number;
  finger_type: string;
  touched_id: number | null;
  struck_id: number | null;
  label: string;
  peak: number | null;
  candidate_score: number | null;
  confidence: number | null;
  confidence_label: string | null;
  strategy: string | null;
}

interface AlternatingSlotStringCandidatePayload {
  string_id?: number;
  count?: number;
  confidence_sum?: number;
  max_confidence?: number;
}

interface AlternatingSlotEntryPayload {
  beat_index?: number;
  beat_time_sec?: number;
  strings?: number[];
  left_hand_involved?: boolean;
  left_hand_note?: string;
  left_hand_touch_count_near_slot?: number;
  string_candidates?: AlternatingSlotStringCandidatePayload[];
}

interface AlternatingOnOffSlotsPayload {
  format?: string;
  slot_start?: 'on_beat' | 'off_beat' | string;
  sequence_length?: number;
  sequence?: Array<Record<string, AlternatingSlotEntryPayload>>;
}

interface PredictionDebugSummary {
  predictedVideoUrlRaw: string | null;
  framesProcessed: number | null;
  strikeResultsCount: number;
  rightDecisionEventsJsonPath: string | null;
  rightStrikeEventsJsonPath: string | null;
  rightAudioDecisionEventsJsonPath: string | null;
  rightAudioStrikeEventsJsonPath: string | null;
  rightAvDecisionEventsJsonPath: string | null;
  rightAvStrikeEventsJsonPath: string | null;
  rightAudioDecisionEventsCount: number;
  rightAudioStrikeEventsCount: number;
  rightAvDecisionEventsCount: number;
  rightAvStrikeEventsCount: number;
  audioPostprocessError: string | null;
  audioExtractionError: string | null;
  fusionMode: string | null;
  audioDecisionMode: string | null;
  analysisDebugReportUrl: string | null;
  analysisDebugSnapshotCount: number;
  analysisDebugFlagCounts: Record<string, number>;
  responseKeys: string[];
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.PROD ? '' : 'http://127.0.0.1:8000');

const EVENT_ACTIVE_WINDOW_SEC = 0.2;
const DISPLAY_FINGER_TYPES = new Set(['thumb', 'index']);
const FAST_MODE_STRING_INFER_EVERY_N = 4;
const FAST_MODE_MAX_STRIKE_EVENTS = 80;
const FAST_MODE_STRIKE_MIN_EVENT_GAP_FRAMES = 12;
const ACCURATE_MODE_STRING_INFER_EVERY_N = 1;
const ACCURATE_MODE_MAX_STRIKE_EVENTS = 180;
const ACCURATE_MODE_STRIKE_MIN_EVENT_GAP_FRAMES = 6;
const ACCURATE_MODE_AUDIO_DECISION_MODE = 'onset_pitch_match';
const FAST_MODE_AUDIO_DECISION_MODE = 'onset_only';
const ACTIVE_PREDICTION_JOB_STORAGE_KEY = 'myanmar-harp-active-prediction-job';

const analyzerSteps = [
  { icon: UploadCloud, title: 'Choose a video', text: 'Upload an MP4, MOV, or WEBM performance clip.' },
  { icon: Zap, title: 'Run analysis', text: 'The app detects hands, strings, touch events, and audio timing.' },
  { icon: CheckCircle2, title: 'Review output', text: 'Check the annotated video, timeline rows, and generated notes.' },
];

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));

const formatSeconds = (value: number) => {
  const safe = Number.isFinite(value) ? Math.max(0, value) : 0;
  const minutes = Math.floor(safe / 60);
  const seconds = safe - minutes * 60;
  return `${minutes}:${seconds.toFixed(2).padStart(5, '0')}`;
};

const Index = () => {
  const [status, setStatus] = useState<ProcessingStatus>('idle');
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [predictedVideoUrl, setPredictedVideoUrl] = useState<string | null>(null);
  const [ksyNotes, setKsyNotes] = useState<string[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [videoLoadError, setVideoLoadError] = useState<string | null>(null);
  const [outputCodec, setOutputCodec] = useState<string | null>(null);
  const [inputHasAudioTrack, setInputHasAudioTrack] = useState<boolean | null>(null);
  const [hasAudioTrack, setHasAudioTrack] = useState<boolean | null>(null);
  const [touchEvents, setTouchEvents] = useState<TouchMappingPayload[]>([]);
  const [strikeResults, setStrikeResults] = useState<StrikeResultPayload[]>([]);
  const [audioDecisionEvents, setAudioDecisionEvents] = useState<AudioDecisionEventPayload[]>([]);
  const [avStrikeEvents, setAvStrikeEvents] = useState<AvStrikeEventPayload[]>([]);
  const [alternatingOnOffSlots, setAlternatingOnOffSlots] = useState<AlternatingOnOffSlotsPayload | null>(null);
  const [processingMode, setProcessingMode] = useState<ProcessingMode>('fast');
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeJobFileName, setActiveJobFileName] = useState<string | null>(null);
  const [currentVideoTime, setCurrentVideoTime] = useState(0);
  const [videoDurationSec, setVideoDurationSec] = useState(0);
  const [selectedEventIndex, setSelectedEventIndex] = useState<number | null>(null);
  const [predictionDebugSummary, setPredictionDebugSummary] = useState<PredictionDebugSummary | null>(null);
  const predictedVideoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    if (!isProcessing) {
      return;
    }

    const stages: ProcessingStatus[] = ['analyzing', 'detecting', 'generating'];
    let stageIndex = 0;
    setStatus(stages[stageIndex]);

    const timer = setInterval(() => {
      stageIndex = Math.min(stageIndex + 1, stages.length - 1);
      setStatus(stages[stageIndex]);
    }, 2200);

    return () => clearInterval(timer);
  }, [isProcessing]);

  const resolveMediaUrl = (relativeOrAbsoluteUrl: string, apiResponseUrl?: string) => {
    if (relativeOrAbsoluteUrl.startsWith('http://') || relativeOrAbsoluteUrl.startsWith('https://')) {
      return relativeOrAbsoluteUrl;
    }

    if (apiResponseUrl) {
      try {
        return new URL(relativeOrAbsoluteUrl, apiResponseUrl).toString();
      } catch {
        // fall through to base URL strategy
      }
    }

    return `${API_BASE_URL}${relativeOrAbsoluteUrl}`;
  };

  const clearActivePredictionJob = () => {
    localStorage.removeItem(ACTIVE_PREDICTION_JOB_STORAGE_KEY);
    setActiveJobId(null);
    setActiveJobFileName(null);
  };

  const storeActivePredictionJob = (jobId: string, fileName: string | null) => {
    localStorage.setItem(
      ACTIVE_PREDICTION_JOB_STORAGE_KEY,
      JSON.stringify({
        jobId,
        fileName,
        startedAt: Date.now(),
      }),
    );
    setActiveJobId(jobId);
    setActiveJobFileName(fileName);
  };

  const resetPredictionOutput = () => {
    setPredictedVideoUrl(null);
    setKsyNotes([]);
    setErrorMessage(null);
    setVideoLoadError(null);
    setOutputCodec(null);
    setInputHasAudioTrack(null);
    setHasAudioTrack(null);
    setTouchEvents([]);
    setStrikeResults([]);
    setAudioDecisionEvents([]);
    setAvStrikeEvents([]);
    setAlternatingOnOffSlots(null);
    setCurrentVideoTime(0);
    setVideoDurationSec(0);
    setSelectedEventIndex(null);
    setPredictionDebugSummary(null);
  };

  const applyPredictionPayload = (payload: PredictionResponse, apiResponseUrl?: string) => {
    const predictedVideoUrlRaw =
      typeof payload.predicted_video_url === 'string' && payload.predicted_video_url.trim().length > 0
        ? payload.predicted_video_url
        : null;
    setPredictedVideoUrl(
      predictedVideoUrlRaw ? resolveMediaUrl(predictedVideoUrlRaw, apiResponseUrl) : null,
    );
    setKsyNotes(Array.isArray(payload.ksy_notes) ? payload.ksy_notes : []);
    setOutputCodec(payload.final_codec ?? null);
    setInputHasAudioTrack(typeof payload.input_has_audio_track === 'boolean' ? payload.input_has_audio_track : null);
    setHasAudioTrack(typeof payload.has_audio_track === 'boolean' ? payload.has_audio_track : null);
    setTouchEvents(Array.isArray(payload.touch_events) ? payload.touch_events : []);
    setStrikeResults(Array.isArray(payload.strike_results) ? payload.strike_results : []);
    setAudioDecisionEvents(
      Array.isArray(payload.right_audio_decision_events) ? payload.right_audio_decision_events : [],
    );
    setAvStrikeEvents(Array.isArray(payload.right_av_strike_events) ? payload.right_av_strike_events : []);
    setAlternatingOnOffSlots(
      payload.right_av_alternating_on_off_slots && typeof payload.right_av_alternating_on_off_slots === 'object'
        ? payload.right_av_alternating_on_off_slots
        : null,
    );
    setPredictionDebugSummary({
      predictedVideoUrlRaw,
      framesProcessed:
        typeof payload.frames_processed === 'number' && Number.isFinite(payload.frames_processed)
          ? payload.frames_processed
          : null,
      strikeResultsCount: Array.isArray(payload.strike_results) ? payload.strike_results.length : 0,
      rightDecisionEventsJsonPath:
        typeof payload.right_decision_events_json_path === 'string'
          ? payload.right_decision_events_json_path
          : null,
      rightStrikeEventsJsonPath:
        typeof payload.right_strike_events_json_path === 'string'
          ? payload.right_strike_events_json_path
          : null,
      rightAudioDecisionEventsJsonPath:
        typeof payload.right_audio_decision_events_json_path === 'string'
          ? payload.right_audio_decision_events_json_path
          : null,
      rightAudioStrikeEventsJsonPath:
        typeof payload.right_audio_strike_events_json_path === 'string'
          ? payload.right_audio_strike_events_json_path
          : null,
      rightAvDecisionEventsJsonPath:
        typeof payload.right_av_decision_events_json_path === 'string'
          ? payload.right_av_decision_events_json_path
          : null,
      rightAvStrikeEventsJsonPath:
        typeof payload.right_av_strike_events_json_path === 'string'
          ? payload.right_av_strike_events_json_path
          : null,
      rightAudioDecisionEventsCount:
        Array.isArray(payload.right_audio_decision_events)
          ? payload.right_audio_decision_events.length
          : typeof payload.right_audio_decision_events_count === 'number'
            ? payload.right_audio_decision_events_count
            : 0,
      rightAudioStrikeEventsCount:
        typeof payload.right_audio_strike_events_count === 'number' ? payload.right_audio_strike_events_count : 0,
      rightAvDecisionEventsCount:
        typeof payload.right_av_decision_events_count === 'number' ? payload.right_av_decision_events_count : 0,
      rightAvStrikeEventsCount:
        typeof payload.right_av_strike_events_count === 'number' ? payload.right_av_strike_events_count : 0,
      audioPostprocessError:
        payload.av_inference && typeof payload.av_inference === 'object' && typeof payload.av_inference.error === 'string'
          ? payload.av_inference.error
          : null,
      audioExtractionError:
        payload.av_inference && typeof payload.av_inference === 'object' && typeof payload.av_inference.audio_error === 'string'
          ? payload.av_inference.audio_error
          : null,
      fusionMode:
        payload.av_inference && typeof payload.av_inference === 'object' && typeof payload.av_inference.fusion_mode === 'string'
          ? payload.av_inference.fusion_mode
          : null,
      audioDecisionMode:
        payload.av_inference && typeof payload.av_inference === 'object' && typeof payload.av_inference.audio_decision_mode === 'string'
          ? payload.av_inference.audio_decision_mode
          : payload.run_profile && typeof payload.run_profile === 'object' && typeof payload.run_profile.audio_decision_mode === 'string'
            ? payload.run_profile.audio_decision_mode
            : null,
      analysisDebugReportUrl:
        typeof payload.analysis_debug_report_url === 'string' && payload.analysis_debug_report_url.length > 0
          ? payload.analysis_debug_report_url
          : null,
      analysisDebugSnapshotCount:
        typeof payload.analysis_debug_snapshot_count === 'number' ? payload.analysis_debug_snapshot_count : 0,
      analysisDebugFlagCounts:
        payload.analysis_debug_summary &&
        typeof payload.analysis_debug_summary === 'object' &&
        payload.analysis_debug_summary.flag_counts &&
        typeof payload.analysis_debug_summary.flag_counts === 'object'
          ? payload.analysis_debug_summary.flag_counts
          : {},
      responseKeys:
        payload && typeof payload === 'object'
          ? Object.keys(payload as Record<string, unknown>).sort()
          : [],
    });
    clearActivePredictionJob();
    setIsProcessing(false);
    setStatus('complete');
  };

  const responseErrorDetail = async (response: Response, fallback: string) => {
    try {
      const errorPayload = await response.json();
      if (typeof errorPayload?.detail === 'string') {
        return errorPayload.detail;
      }
    } catch {
      // Keep fallback message.
    }
    return fallback;
  };

  const handlePredictionJobUpdate = (job: PredictionJobResponse, apiResponseUrl?: string) => {
    const jobId = typeof job.job_id === 'string' ? job.job_id : activeJobId;
    const fileName = typeof job.file_name === 'string' ? job.file_name : activeJobFileName;
    const jobStatus = String(job.status ?? '').toLowerCase();

    if (jobId && jobStatus !== 'complete' && jobStatus !== 'failed') {
      storeActivePredictionJob(jobId, fileName ?? null);
      setIsProcessing(true);
      setStatus((current) => (current === 'idle' ? 'analyzing' : current));
      return;
    }

    if (jobStatus === 'complete' && job.result && typeof job.result === 'object') {
      applyPredictionPayload(job.result, apiResponseUrl);
      return;
    }

    if (jobStatus === 'failed') {
      clearActivePredictionJob();
      setIsProcessing(false);
      setStatus('idle');
      setErrorMessage(job.error || 'Prediction failed.');
    }
  };

  useEffect(() => {
    try {
      const raw = localStorage.getItem(ACTIVE_PREDICTION_JOB_STORAGE_KEY);
      if (!raw) {
        return;
      }
      const parsed = JSON.parse(raw) as { jobId?: string; fileName?: string };
      if (typeof parsed.jobId !== 'string' || parsed.jobId.length === 0) {
        localStorage.removeItem(ACTIVE_PREDICTION_JOB_STORAGE_KEY);
        return;
      }
      setActiveJobId(parsed.jobId);
      setActiveJobFileName(typeof parsed.fileName === 'string' ? parsed.fileName : null);
      setIsProcessing(true);
      setStatus('analyzing');
    } catch {
      localStorage.removeItem(ACTIVE_PREDICTION_JOB_STORAGE_KEY);
    }
  }, []);

  useEffect(() => {
    if (!activeJobId) {
      return;
    }

    let cancelled = false;
    const pollJob = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/jobs/${activeJobId}`);
        if (!response.ok) {
          throw new Error(await responseErrorDetail(response, 'Prediction job could not be found.'));
        }
        const job: PredictionJobResponse = await response.json();
        if (!cancelled) {
          handlePredictionJobUpdate(job, response.url);
        }
      } catch (error) {
        if (cancelled) {
          return;
        }
        clearActivePredictionJob();
        setIsProcessing(false);
        setStatus('idle');
        setErrorMessage(error instanceof Error ? error.message : 'Prediction job polling failed.');
      }
    };

    pollJob();
    const timer = setInterval(pollJob, 2500);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [activeJobId]);

  const handleUpload = async (file: File) => {
    setVideoFile(file);
    resetPredictionOutput();
    setIsProcessing(true);
    setStatus('analyzing');

    try {
      const query = new URLSearchParams();
      const isFastMode = processingMode === 'fast';
      query.set('fast_mode', String(isFastMode));
      query.set('audio_enabled', 'true');
      query.set('fusion_mode', 'av_fuse');
      query.set('enable_debug_report', 'true');
      query.set('include_strike_debug', String(!isFastMode));
      if (isFastMode) {
        query.set('string_infer_every_n', String(FAST_MODE_STRING_INFER_EVERY_N));
        query.set('max_strike_events', String(FAST_MODE_MAX_STRIKE_EVENTS));
        query.set('strike_min_event_gap_frames', String(FAST_MODE_STRIKE_MIN_EVENT_GAP_FRAMES));
        query.set('audio_decision_mode', FAST_MODE_AUDIO_DECISION_MODE);
      } else {
        query.set('string_infer_every_n', String(ACCURATE_MODE_STRING_INFER_EVERY_N));
        query.set('max_strike_events', String(ACCURATE_MODE_MAX_STRIKE_EVENTS));
        query.set('strike_min_event_gap_frames', String(ACCURATE_MODE_STRIKE_MIN_EVENT_GAP_FRAMES));
        query.set('audio_decision_mode', ACCURATE_MODE_AUDIO_DECISION_MODE);
      }
      const predictUrl = `${API_BASE_URL}/api/jobs/predict-video${query.toString() ? `?${query.toString()}` : ''}`;

      const response = await fetch(predictUrl, {
        method: 'POST',
        headers: {
          'Content-Type': file.type || 'application/octet-stream',
          'X-File-Name': file.name,
        },
        body: file,
      });

      if (!response.ok) {
        throw new Error(await responseErrorDetail(response, 'Prediction failed.'));
      }

      const job: PredictionJobResponse = await response.json();
      handlePredictionJobUpdate(job, response.url);
    } catch (error) {
      clearActivePredictionJob();
      const message = error instanceof Error ? error.message : 'Prediction failed.';
      setErrorMessage(message);
      setStatus('idle');
      setIsProcessing(false);
    }
  };

  const sortedStrikeResults = [...strikeResults].sort(
    (a, b) => (Number(a.event_time) || 0) - (Number(b.event_time) || 0),
  );
  const vibrationEvents = sortedStrikeResults.filter((event) => {
    const label = String(event.label ?? '').toLowerCase();
    return Boolean(event.best_metrics?.vibrates) || label === 'strike';
  });
  const sortedAvStrikeEvents = [...avStrikeEvents].sort(
    (a, b) => (Number(a.timestamp_sec) || 0) - (Number(b.timestamp_sec) || 0),
  );
  const timelineEvents: TimelineEventPayload[] =
    sortedAvStrikeEvents.length > 0
      ? sortedAvStrikeEvents.map((event) => ({
          source: 'av',
          event_time: Number(event.timestamp_sec) || 0,
          finger_type: String(event.finger_type ?? '-'),
          touched_id:
            typeof event.touched_string_id === 'number' && Number.isFinite(event.touched_string_id)
              ? event.touched_string_id
              : null,
          struck_id:
            typeof event.struck_string_id === 'number' && Number.isFinite(event.struck_string_id)
              ? event.struck_string_id
              : null,
          label: 'strike',
          peak: null,
          candidate_score: null,
          confidence:
            typeof event.confidence === 'number' && Number.isFinite(event.confidence) ? event.confidence : null,
          confidence_label:
            typeof event.confidence_label === 'string' && event.confidence_label.trim().length > 0
              ? event.confidence_label
              : null,
          strategy:
            typeof event.strategy === 'string' && event.strategy.trim().length > 0 ? event.strategy : null,
        }))
      : vibrationEvents.map((event) => ({
          source: 'vibration',
          event_time: Number(event.event_time) || 0,
          finger_type: String(event.finger_type ?? '-'),
          touched_id:
            typeof event.touched_id === 'number' && Number.isFinite(event.touched_id) ? event.touched_id : null,
          struck_id:
            typeof event.struck_id === 'number' && Number.isFinite(event.struck_id) ? event.struck_id : null,
          label: String(event.label ?? 'unknown'),
          peak:
            typeof event.best_metrics?.peak === 'number' && Number.isFinite(event.best_metrics.peak)
              ? event.best_metrics.peak
              : null,
          candidate_score:
            typeof event.best_metrics?.candidate_score === 'number' &&
            Number.isFinite(event.best_metrics.candidate_score)
              ? event.best_metrics.candidate_score
              : null,
          confidence: null,
          confidence_label: null,
          strategy: null,
        }));
  const usingAvStrikeTimeline = sortedAvStrikeEvents.length > 0;
  const timelineDuration =
    videoDurationSec > 0
      ? videoDurationSec
      : Math.max(0, ...timelineEvents.map((event) => Number(event.event_time) || 0));

  let autoActiveEventIndex: number | null = null;
  let autoActiveDelta = Number.POSITIVE_INFINITY;
  timelineEvents.forEach((event, index) => {
    const delta = Math.abs((Number(event.event_time) || 0) - currentVideoTime);
    if (delta <= EVENT_ACTIVE_WINDOW_SEC && delta < autoActiveDelta) {
      autoActiveDelta = delta;
      autoActiveEventIndex = index;
    }
  });

  const activeEventIndex = selectedEventIndex ?? autoActiveEventIndex;
  const activeEvent =
    activeEventIndex !== null && activeEventIndex >= 0 && activeEventIndex < timelineEvents.length
      ? timelineEvents[activeEventIndex]
      : null;

  const sortedAudioDecisionEvents = [...audioDecisionEvents].sort((a, b) => {
    const ta = Number(a.touch?.timestamp_sec ?? a.audio?.touch_time_sec ?? 0) || 0;
    const tb = Number(b.touch?.timestamp_sec ?? b.audio?.touch_time_sec ?? 0) || 0;
    return ta - tb;
  });

  const visibleAudioDecisionEvents = sortedAudioDecisionEvents.filter((event) =>
    DISPLAY_FINGER_TYPES.has(String(event.touch?.finger_type ?? '').toLowerCase()),
  );

  const audioStatusCounts = visibleAudioDecisionEvents.reduce<Record<string, number>>((acc, event) => {
    const status = String(event.audio?.status ?? 'unknown').toLowerCase();
    acc[status] = (acc[status] ?? 0) + 1;
    return acc;
  }, {});

  const touchMappingRows =
    touchEvents.length > 0
      ? [...touchEvents]
          .filter((event) => DISPLAY_FINGER_TYPES.has(String(event.finger_type ?? '').toLowerCase()))
          .sort((a, b) => {
            const ta = Number(a.timestamp_sec ?? a.time_sec ?? 0) || 0;
            const tb = Number(b.timestamp_sec ?? b.time_sec ?? 0) || 0;
            return ta - tb;
          })
          .map((event, index) => ({
            eventId: `touch-${index}-${String(event.hand_side ?? event.hand ?? 'unknown')}-${String(event.finger_type ?? 'x')}`,
            timestampSec: Number(event.timestamp_sec ?? event.time_sec ?? 0) || 0,
            frameIndex:
              typeof event.frame_index === 'number' && Number.isFinite(event.frame_index) ? event.frame_index : null,
            handSide: String(event.hand_side ?? event.hand ?? 'unknown'),
            fingerType: String(event.finger_type ?? '-'),
            touchedStringId:
              typeof (event.touched_string_id ?? event.string_id) === 'number' &&
              Number.isFinite(Number(event.touched_string_id ?? event.string_id))
                ? Number(event.touched_string_id ?? event.string_id)
                : null,
            touchConf:
              typeof event.touch_conf === 'number' && Number.isFinite(event.touch_conf) ? event.touch_conf : null,
            distancePx:
              typeof event.distance_px === 'number' && Number.isFinite(event.distance_px) ? event.distance_px : null,
          }))
      : visibleAudioDecisionEvents.length > 0
      ? visibleAudioDecisionEvents.map((event) => ({
          eventId: event.event_id ?? '',
          timestampSec: Number(event.touch?.timestamp_sec ?? event.audio?.touch_time_sec ?? 0) || 0,
          frameIndex:
            typeof event.touch?.frame_index === 'number' && Number.isFinite(event.touch.frame_index)
              ? event.touch.frame_index
              : null,
          handSide: String(event.touch?.hand_side ?? 'right'),
          fingerType: String(event.touch?.finger_type ?? '-'),
          touchedStringId:
            typeof event.touch?.touched_string_id === 'number' && Number.isFinite(event.touch.touched_string_id)
              ? event.touch.touched_string_id
              : null,
          touchConf:
            typeof event.touch?.touch_conf === 'number' && Number.isFinite(event.touch.touch_conf)
              ? event.touch.touch_conf
              : null,
          distancePx:
            typeof event.touch?.distance_px === 'number' && Number.isFinite(event.touch.distance_px)
              ? event.touch.distance_px
              : null,
        }))
      : ([] as Array<{
          eventId: string;
          timestampSec: number;
          frameIndex: number | null;
          handSide: string;
          fingerType: string;
          touchedStringId: number | null;
          touchConf: number | null;
          distancePx: number | null;
        }>);

  return (
    <div className="min-h-screen relative overflow-hidden">
      <AnimatedBackground />
      <SiteNav />
      
      <div className="relative z-10 pb-16 pt-24 md:pb-24">
        <main id="analyzer" className="container mx-auto px-4 space-y-12">
          <section className="mx-auto max-w-6xl">
            <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
              <div>
                <div className="inline-flex items-center gap-2 rounded-lg border border-cyan-300/25 bg-cyan-300/10 px-3 py-1.5 text-xs font-medium uppercase tracking-[0.18em] text-cyan-200">
                  <Settings2 className="h-3.5 w-3.5" />
                  Analyzer
                </div>
                <h2 className="mt-3 font-heading text-3xl font-bold leading-tight text-white md:text-4xl">
                  Analyze your harp video
                </h2>
                <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">
                  Upload a performance clip and the app will prepare the annotated video, note events, and review timeline.
                </p>
              </div>
              <div className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-white/55">
                Supported: MP4, MOV, WEBM
              </div>
            </div>

            {!(status === 'complete' && predictedVideoUrl) && (
            <div className="rounded-2xl border border-white/10 bg-card/70 p-4 shadow-[0_18px_70px_rgba(0,0,0,0.22)] backdrop-blur-xl">
              <div className="grid items-stretch gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
                <div className="h-full">
                  {status === 'idle' && (
                    <VideoUploadZone
                      onUpload={handleUpload}
                      isProcessing={isProcessing}
                    />
                  )}

                  {isProcessing && (
                    <div className="flex h-full items-center justify-center rounded-xl border border-white/10 bg-black/20 p-6 animate-scale-in">
                      <SoundWaveLoader
                        status={status as 'analyzing' | 'detecting' | 'generating'}
                      />
                    </div>
                  )}
                </div>

                <aside className="flex h-full flex-col gap-3">
                  <div className="rounded-xl border border-white/10 bg-black/20 p-3">
                    <div className="space-y-3">
                      <div>
                        <div className="flex items-center gap-2">
                          <Gauge className="h-4 w-4 text-emerald-300" />
                          <h3 className="text-sm font-semibold text-white">Processing mode</h3>
                        </div>
                        <p className="mt-1.5 text-xs leading-5 text-white/55">
                          {processingMode === 'fast'
                            ? `Fast: YOLO every ${FAST_MODE_STRING_INFER_EVERY_N} frames, ${FAST_MODE_MAX_STRIKE_EVENTS} strike checks.`
                            : `Accurate: YOLO every frame, ${ACCURATE_MODE_MAX_STRIKE_EVENTS} strike checks, pitch matching.`}
                        </p>
                      </div>
                      <div className="grid grid-cols-2 rounded-lg border border-white/10 bg-black/25 p-1">
                        {(['fast', 'accurate'] as ProcessingMode[]).map((mode) => (
                          <button
                            key={mode}
                            type="button"
                            disabled={isProcessing}
                            onClick={() => setProcessingMode(mode)}
                            className={`rounded-md px-3 py-2 text-xs font-medium capitalize transition-colors ${
                              processingMode === mode
                                ? 'bg-cyan-300/20 text-cyan-100 ring-1 ring-cyan-300/35'
                                : 'text-white/55 hover:bg-white/5 hover:text-white/80'
                            } ${isProcessing ? 'cursor-not-allowed opacity-60' : ''}`}
                          >
                            {mode}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="rounded-xl border border-white/10 bg-black/20 p-3">
                    <h3 className="text-sm font-semibold text-white">What happens next</h3>
                    <div className="mt-3 space-y-3">
                      {analyzerSteps.map((step, index) => {
                        const Icon = step.icon;
                        return (
                          <div key={step.title} className="flex gap-3">
                            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-white/10 bg-white/10">
                              <Icon className="h-4 w-4 text-cyan-200" />
                            </div>
                            <div>
                              <p className="text-sm font-medium text-white">
                                {index + 1}. {step.title}
                              </p>
                              <p className="mt-0.5 text-xs leading-5 text-white/50">{step.text}</p>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </aside>
              </div>
            </div>
            )}
          </section>

          <div className="space-y-12">

          {errorMessage && (
            <div className="max-w-xl mx-auto glass-strong rounded-2xl p-6 text-center text-destructive animate-fade-in">
              {errorMessage}
            </div>
          )}

          {status === 'complete' && !predictedVideoUrl && (
            <div className="max-w-4xl mx-auto glass-strong rounded-2xl p-6 animate-fade-in">
              <div className="rounded-xl border border-amber-400/30 bg-amber-500/10 p-4">
                <h3 className="font-heading text-lg font-bold text-amber-200">
                  Prediction Finished, But No Video URL Was Returned
                </h3>
                <p className="mt-2 text-sm text-amber-100/90">
                  The backend appears to have finished processing, but the frontend did not receive a valid
                  `predicted_video_url`, so the video player cannot be shown.
                </p>
              </div>

              {(videoFile || activeJobFileName) && (
                <p className="mt-4 text-xs text-muted-foreground truncate">
                  Source: {videoFile?.name ?? activeJobFileName}
                </p>
              )}

              {predictionDebugSummary && (
                <div className="mt-4 grid gap-3 md:grid-cols-2">
                  <div className="rounded-xl border border-border/40 bg-background/20 p-4 text-xs">
                    <p className="font-medium text-foreground/90">Response Diagnostics</p>
                    <p className="mt-2 text-muted-foreground">
                      `predicted_video_url` (raw):{' '}
                      <span className="break-all text-foreground/90">
                        {predictionDebugSummary.predictedVideoUrlRaw ?? 'null / missing'}
                      </span>
                    </p>
                    <p className="mt-1 text-muted-foreground">
                      frames_processed: {predictionDebugSummary.framesProcessed ?? 'unknown'}
                    </p>
                    <p className="mt-1 text-muted-foreground">
                      strike_results_count: {predictionDebugSummary.strikeResultsCount}
                    </p>
                    <p className="mt-2 text-muted-foreground">Response keys:</p>
                    <p className="mt-1 break-all text-[11px] text-foreground/85">
                      {predictionDebugSummary.responseKeys.join(', ') || '(none)'}
                    </p>
                  </div>

                  <div className="rounded-xl border border-border/40 bg-background/20 p-4 text-xs">
                    <p className="font-medium text-foreground/90">Generated Event Files</p>
                    <p className="mt-2 text-muted-foreground">right decision events JSON:</p>
                    <p className="mt-1 break-all text-[11px] text-foreground/85">
                      {predictionDebugSummary.rightDecisionEventsJsonPath ?? 'not provided'}
                    </p>
                    <p className="mt-2 text-muted-foreground">right strike events JSON:</p>
                    <p className="mt-1 break-all text-[11px] text-foreground/85">
                      {predictionDebugSummary.rightStrikeEventsJsonPath ?? 'not provided'}
                    </p>
                    <p className="mt-2 text-muted-foreground">right audio decision events JSON:</p>
                    <p className="mt-1 break-all text-[11px] text-foreground/85">
                      {predictionDebugSummary.rightAudioDecisionEventsJsonPath ?? 'not provided'}
                    </p>
                    <p className="mt-2 text-muted-foreground">right audio strike events JSON:</p>
                    <p className="mt-1 break-all text-[11px] text-foreground/85">
                      {predictionDebugSummary.rightAudioStrikeEventsJsonPath ?? 'not provided'}
                    </p>
                    <p className="mt-2 text-muted-foreground">right AV decision events JSON:</p>
                    <p className="mt-1 break-all text-[11px] text-foreground/85">
                      {predictionDebugSummary.rightAvDecisionEventsJsonPath ?? 'not provided'}
                    </p>
                    <p className="mt-2 text-muted-foreground">right AV strike events JSON:</p>
                    <p className="mt-1 break-all text-[11px] text-foreground/85">
                      {predictionDebugSummary.rightAvStrikeEventsJsonPath ?? 'not provided'}
                    </p>
                    <p className="mt-2 text-muted-foreground">
                      right audio decision events (inline): {predictionDebugSummary.rightAudioDecisionEventsCount}
                    </p>
                  </div>
                </div>
              )}

              <div className="mt-4 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => {
                    clearActivePredictionJob();
                    setStatus('idle');
                    setPredictedVideoUrl(null);
                    setVideoLoadError(null);
                    setAvStrikeEvents([]);
                    setAlternatingOnOffSlots(null);
                  }}
                  className="glass px-4 py-2 rounded-xl text-xs font-medium hover:bg-primary/10 transition-colors duration-300"
                >
                  Back To Upload
                </button>
              </div>
            </div>
          )}

          {/* Predicted Video */}
          {status === 'complete' && predictedVideoUrl && (
            <div className="max-w-6xl mx-auto glass-strong rounded-2xl p-6 animate-fade-in">
              <div className="mb-4">
                <h3 className="font-heading text-lg font-bold">Predicted Video</h3>
                <p className="text-xs text-muted-foreground">
                  Annotated output generated with `harp_pose_v11m_prepped/weights/best.pt`
                </p>
              </div>

              <div className="relative">
                <video
                  ref={predictedVideoRef}
                  src={predictedVideoUrl}
                  className="w-full rounded-xl"
                  controls
                  preload="metadata"
                  onLoadedMetadata={(event) => {
                    setVideoDurationSec(Number(event.currentTarget.duration) || 0);
                    setCurrentVideoTime(Number(event.currentTarget.currentTime) || 0);
                  }}
                  onTimeUpdate={(event) => {
                    setCurrentVideoTime(Number(event.currentTarget.currentTime) || 0);
                  }}
                  onSeeked={(event) => {
                    setCurrentVideoTime(Number(event.currentTarget.currentTime) || 0);
                  }}
                  onPlay={() => {
                    setSelectedEventIndex(null);
                  }}
                  onError={() => {
                    setVideoLoadError(`Unable to play predicted video at: ${predictedVideoUrl}`);
                  }}
                />
                {/* ---------------------------------- Active Event Overlay --
                {activeEvent && (
                  <div className="pointer-events-none absolute left-3 top-3 max-w-[calc(100%-1.5rem)] rounded-xl border border-white/15 bg-black/65 px-3 py-2 text-xs text-white backdrop-blur-sm">
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                      <span className="font-semibold text-cyan-200">{formatSeconds(Number(activeEvent.event_time) || 0)}</span>
                      <span>
                        Vibrated: <span className="font-semibold text-emerald-300">s{activeEvent.struck_id ?? activeEvent.touched_id ?? '?'}</span>
                      </span>
                      <span>
                        Label: <span className="font-semibold">{String(activeEvent.label ?? 'unknown')}</span>
                      </span>
                      <span>
                        Finger: <span className="font-semibold">{String(activeEvent.finger_type ?? '-')}</span>
                      </span>
                    </div>
                    <div className="mt-1 text-[11px] text-white/80">
                      {activeEvent.source === 'av'
                        ? `touched s${activeEvent.touched_id ?? '?'} | conf ${Number(activeEvent.confidence ?? 0).toFixed(2)}${activeEvent.confidence_label ? ` (${activeEvent.confidence_label})` : ''}${activeEvent.strategy ? ` | ${activeEvent.strategy}` : ''}`
                        : `touched s${activeEvent.touched_id ?? '?'} | peak z ${Number(activeEvent.peak ?? 0).toFixed(2)} | score ${Number(activeEvent.candidate_score ?? 0).toFixed(2)}`}
                    </div>
                  </div>
                )}-------------------------------- */}
              </div>

              {timelineEvents.length > 0 && (
                <div className="mt-4 rounded-xl border border-border/40 bg-background/20 p-3">
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <p className="text-xs font-medium text-foreground/90">
                      {usingAvStrikeTimeline ? 'AV Strike Timeline' : 'Vibration Timeline'} ({timelineEvents.length} event{timelineEvents.length === 1 ? '' : 's'})
                    </p>
                    <p className="text-[11px] text-muted-foreground">
                      Click a marker to jump to that event
                    </p>
                  </div>

                  <div className="relative h-10 rounded-lg bg-black/20">
                    <div className="absolute inset-x-2 top-1/2 h-px -translate-y-1/2 bg-white/20" />
                    {timelineDuration > 0 && (
                      <div
                        className="absolute bottom-1 top-1 w-px bg-cyan-300/80"
                        style={{ left: `calc(${clamp01(currentVideoTime / timelineDuration) * 100}% - 0.5px)` }}
                      />
                    )}
                    {timelineEvents.map((event, index) => {
                      const t = Number(event.event_time) || 0;
                      const label = String(event.label ?? '').toLowerCase();
                      const isStrike = label === 'strike';
                      const isActive = activeEventIndex === index;
                      const leftPct = timelineDuration > 0 ? clamp01(t / timelineDuration) * 100 : 0;
                      return (
                        <button
                          key={`${t}-${event.struck_id ?? event.touched_id ?? index}-${index}`}
                          type="button"
                          aria-label={`Jump to event at ${formatSeconds(t)}`}
                          title={`${formatSeconds(t)} | vibrated s${event.struck_id ?? event.touched_id ?? '?'} | ${label || 'unknown'}`}
                          className={`absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border transition-all ${
                            isStrike
                              ? 'border-emerald-300 bg-emerald-400/80'
                              : 'border-amber-300 bg-amber-400/75'
                          } ${isActive ? 'scale-125 ring-2 ring-white/70' : 'hover:scale-110'}`}
                          style={{ left: `${leftPct}%` }}
                          onClick={() => {
                            setSelectedEventIndex(index);
                            const video = predictedVideoRef.current;
                            if (video) {
                              video.currentTime = t;
                              setCurrentVideoTime(t);
                              video.pause();
                            }
                          }}
                        />
                      );
                    })}
                  </div>

                  <div className="mt-3 max-h-40 space-y-2 overflow-y-auto pr-1">
                    {timelineEvents.slice(0, 120).map((event, index) => {
                      const label = String(event.label ?? '').toLowerCase();
                      const isActive = activeEventIndex === index;
                      const isStrike = label === 'strike';
                      return (
                        <button
                          key={`row-${index}-${event.event_time ?? index}`}
                          type="button"
                          onClick={() => {
                            setSelectedEventIndex(index);
                            const t = Number(event.event_time) || 0;
                            const video = predictedVideoRef.current;
                            if (video) {
                              video.currentTime = t;
                              setCurrentVideoTime(t);
                              video.pause();
                            }
                          }}
                          className={`flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-xs transition-colors ${
                            isActive ? 'bg-primary/15 ring-1 ring-primary/40' : 'bg-black/10 hover:bg-black/20'
                          }`}
                        >
                          <span className="flex items-center gap-2">
                            <span className={`h-2 w-2 rounded-full ${isStrike ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                            <span className="font-medium">{formatSeconds(Number(event.event_time) || 0)}</span>
                            <span className="text-muted-foreground">
                              vibrated s{event.struck_id ?? event.touched_id ?? '?'} (touched s{event.touched_id ?? '?'})
                            </span>
                          </span>
                          <span className="text-muted-foreground">
                            {event.source === 'av'
                              ? `${label || 'unknown'} | conf ${Number(event.confidence ?? 0).toFixed(2)}${event.confidence_label ? ` (${event.confidence_label})` : ''}`
                              : `${label || 'unknown'} | z ${Number(event.peak ?? 0).toFixed(1)}`}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {predictionDebugSummary && (
                <div className="mt-4 rounded-xl border border-border/40 bg-background/20 p-3 text-xs">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <p className="font-medium text-foreground/90">Audio Mapping Diagnostics</p>
                    <div className="flex flex-wrap gap-2 text-[11px] text-muted-foreground">
                      <span className="rounded-full border border-border/40 bg-black/10 px-2 py-1">
                        audio decisions: {predictionDebugSummary.rightAudioDecisionEventsCount}
                      </span>
                      <span className="rounded-full border border-border/40 bg-black/10 px-2 py-1">
                        audio strikes: {predictionDebugSummary.rightAudioStrikeEventsCount}
                      </span>
                      <span className="rounded-full border border-border/40 bg-black/10 px-2 py-1">
                        av strikes: {predictionDebugSummary.rightAvStrikeEventsCount}
                      </span>
                      <span className="rounded-full border border-border/40 bg-black/10 px-2 py-1">
                        mode: {predictionDebugSummary.fusionMode ?? 'unknown'}
                      </span>
                      <span className="rounded-full border border-border/40 bg-black/10 px-2 py-1">
                        audio: {predictionDebugSummary.audioDecisionMode ?? 'unknown'}
                      </span>
                      <span className="rounded-full border border-border/40 bg-black/10 px-2 py-1">
                        snapshots: {predictionDebugSummary.analysisDebugSnapshotCount}
                      </span>
                    </div>
                  </div>

                  {predictionDebugSummary.analysisDebugReportUrl && (
                    <div className="mt-3 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-cyan-300/20 bg-cyan-400/10 p-3">
                      <div>
                        <p className="text-cyan-100">Debug report generated</p>
                        <p className="mt-1 text-[11px] text-cyan-100/70">
                          Includes conflict rows, pitch failures, timing, and frame snapshots.
                        </p>
                      </div>
                      <a
                        href={predictionDebugSummary.analysisDebugReportUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="rounded-lg border border-cyan-300/30 bg-cyan-300/10 px-3 py-2 text-[11px] font-medium text-cyan-100 transition-colors hover:bg-cyan-300/20"
                      >
                        Open Report
                      </a>
                    </div>
                  )}

                  {Object.keys(predictionDebugSummary.analysisDebugFlagCounts).length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-muted-foreground">
                      {Object.entries(predictionDebugSummary.analysisDebugFlagCounts)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 8)
                        .map(([flag, count]) => (
                          <span key={flag} className="rounded-full border border-border/40 bg-black/10 px-2 py-1">
                            {flag.replace(/_/g, ' ')}: {count}
                          </span>
                        ))}
                    </div>
                  )}

                  {(predictionDebugSummary.audioPostprocessError || predictionDebugSummary.audioExtractionError) ? (
                    <div className="mt-3 rounded-lg border border-amber-300/30 bg-amber-500/10 p-3">
                      <p className="text-amber-100">
                        Audio mapping did not fully run for this response.
                      </p>
                      {predictionDebugSummary.audioPostprocessError && (
                        <p className="mt-1 break-all text-[11px] text-amber-100/90">
                          postprocess error: {predictionDebugSummary.audioPostprocessError}
                        </p>
                      )}
                      {predictionDebugSummary.audioExtractionError && (
                        <p className="mt-1 break-all text-[11px] text-amber-100/90">
                          audio error: {predictionDebugSummary.audioExtractionError}
                        </p>
                      )}
                    </div>
                  ) : predictionDebugSummary.rightAudioDecisionEventsCount === 0 ? (
                    <div className="mt-3 rounded-lg border border-rose-300/20 bg-rose-500/10 p-3 text-rose-100">
                      No inline audio mapping rows were returned by the backend for this upload.
                    </div>
                  ) : (
                    <div className="mt-3 rounded-lg border border-emerald-300/20 bg-emerald-500/10 p-3 text-emerald-100">
                      Audio mapping rows were returned by the backend and should appear below.
                    </div>
                  )}
                </div>
              )}

              {touchMappingRows.length > 0 && (
                <div className="mt-4 rounded-xl border border-border/40 bg-background/20 p-3">
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <p className="text-xs font-medium text-foreground/90">
                      Touch Mapping Results ({touchMappingRows.length})
                    </p>
                    <p className="text-[11px] text-muted-foreground">
                      Showing only right thumb/index and left thumb touch events
                    </p>
                  </div>

                  <div className="max-h-44 space-y-2 overflow-y-auto pr-1">
                    {touchMappingRows.slice(0, 160).map((row, index) => {
                      const isNearTime = Math.abs(row.timestampSec - currentVideoTime) <= EVENT_ACTIVE_WINDOW_SEC;
                      return (
                        <button
                          key={`touch-row-${row.eventId || index}`}
                          type="button"
                          onClick={() => {
                            const video = predictedVideoRef.current;
                            if (video) {
                              video.currentTime = row.timestampSec;
                              setCurrentVideoTime(row.timestampSec);
                              video.pause();
                            }
                          }}
                          className={`flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-xs transition-colors ${
                            isNearTime ? 'bg-cyan-500/10 ring-1 ring-cyan-400/30' : 'bg-black/10 hover:bg-black/20'
                          }`}
                          title={`Jump to touch event at ${formatSeconds(row.timestampSec)}`}
                        >
                          <span className="flex items-center gap-2">
                            <span className="h-2 w-2 rounded-full bg-cyan-300" />
                            <span className="font-medium">{formatSeconds(row.timestampSec)}</span>
                            <span className="text-muted-foreground">
                              {row.handSide} {row.fingerType} touched s{row.touchedStringId ?? '?'}
                            </span>
                          </span>
                          <span className="text-muted-foreground">
                            f{row.frameIndex ?? '?'} | conf {row.touchConf?.toFixed(2) ?? '-'} | d {row.distancePx?.toFixed(1) ?? '-'}px
                          </span>
                        </button>
                      );
                    })}
                    {touchMappingRows.length > 160 && (
                      <p className="px-1 text-[11px] text-muted-foreground">
                        Showing first 160 touch mapping rows.
                      </p>
                    )}
                  </div>
                </div>
              )}

              {visibleAudioDecisionEvents.length > 0 && (
                <div className="mt-4 rounded-xl border border-border/40 bg-background/20 p-3">
                  <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="text-xs font-medium text-foreground/90">
                        Audio Mapping Results ({visibleAudioDecisionEvents.length})
                      </p>
                      <p className="text-[11px] text-muted-foreground">
                        Audio onset-based decision for right thumb/index only
                      </p>
                    </div>
                    <div className="flex flex-wrap gap-2 text-[11px] text-muted-foreground">
                      {Object.entries(audioStatusCounts).slice(0, 8).map(([status, count]) => (
                        <span key={`audio-status-${status}`} className="rounded-full border border-border/40 bg-black/10 px-2 py-1">
                          {status}: {count}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="max-h-52 space-y-2 overflow-y-auto pr-1">
                    {visibleAudioDecisionEvents.slice(0, 160).map((event, index) => {
                      const touchTime = Number(event.touch?.timestamp_sec ?? event.audio?.touch_time_sec ?? 0) || 0;
                      const status = String(event.audio?.status ?? 'unknown').toLowerCase();
                      const isStrike = status === 'strike';
                      const isNearTime = Math.abs(touchTime - currentVideoTime) <= EVENT_ACTIVE_WINDOW_SEC;
                      const touchedStringId = event.touch?.touched_string_id ?? null;
                      const decidedStringId = event.decision?.struck_string_id ?? event.audio?.matched_string_id ?? null;
                      const confidence = Number(event.decision?.confidence ?? 0) || 0;
                      const onsetScore = event.audio?.onset_score;
                      const badgeClass = isStrike
                        ? 'border-emerald-300/40 bg-emerald-500/10 text-emerald-200'
                        : status === 'no_audio'
                          ? 'border-rose-300/30 bg-rose-500/10 text-rose-200'
                          : 'border-amber-300/30 bg-amber-500/10 text-amber-200';
                      return (
                        <button
                          key={`audio-row-${event.event_id ?? index}`}
                          type="button"
                          onClick={() => {
                            const video = predictedVideoRef.current;
                            if (video) {
                              video.currentTime = touchTime;
                              setCurrentVideoTime(touchTime);
                              video.pause();
                            }
                          }}
                          className={`flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-xs transition-colors ${
                            isNearTime ? 'bg-primary/15 ring-1 ring-primary/35' : 'bg-black/10 hover:bg-black/20'
                          }`}
                          title={`Jump to audio mapping event at ${formatSeconds(touchTime)}`}
                        >
                          <span className="flex min-w-0 items-center gap-2">
                            <span className={`h-2 w-2 rounded-full ${isStrike ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                            <span className="font-medium">{formatSeconds(touchTime)}</span>
                            <span className="text-muted-foreground truncate">
                              touch s{touchedStringId ?? '?'} {'->'} audio {isStrike ? `s${decidedStringId ?? '?'}` : 'no strike'}
                            </span>
                            <span className={`rounded-full border px-2 py-0.5 ${badgeClass}`}>
                              {status}
                            </span>
                          </span>
                          <span className="text-muted-foreground">
                            onset {typeof onsetScore === 'number' ? onsetScore.toFixed(2) : '-'} | conf {confidence.toFixed(2)}
                          </span>
                        </button>
                      );
                    })}
                    {visibleAudioDecisionEvents.length > 160 && (
                      <p className="px-1 text-[11px] text-muted-foreground">
                        Showing first 160 audio mapping rows.
                      </p>
                    )}
                  </div>
                </div>
              )}

              {(videoFile || activeJobFileName) && (
                <p className="mt-3 text-xs text-muted-foreground truncate">
                  Source: {videoFile?.name ?? activeJobFileName}
                </p>
              )}

              {videoLoadError && (
                <p className="mt-2 text-xs text-destructive break-all">
                  {videoLoadError}
                </p>
              )}

              {(outputCodec || inputHasAudioTrack !== null || hasAudioTrack !== null) && (
                <p className="mt-2 text-xs text-muted-foreground/80">
                  Output codec: {outputCodec ?? 'unknown'} | Input audio: {inputHasAudioTrack === null ? 'unknown' : inputHasAudioTrack ? 'yes' : 'no'} | Output audio: {hasAudioTrack === null ? 'unknown' : hasAudioTrack ? 'yes' : 'no'}
                </p>
              )}

              <p className="mt-2 text-xs text-muted-foreground/80">
                ksy_notes output is preserved for UI integration ({ksyNotes.length} item{ksyNotes.length === 1 ? '' : 's'}) | raw touch events: {touchEvents.length}
              </p>
            </div>
          )}

          {/* Results */}
          <MusicalNotesDisplay isVisible={status === 'complete'} alternatingSlots={alternatingOnOffSlots} />

          {/* Reset button when complete */}
          {status === 'complete' && (
            <div className="text-center animate-fade-in" style={{ animationDelay: '1s' }}>
              <button
                onClick={() => {
                  clearActivePredictionJob();
                  setStatus('idle');
                  setVideoFile(null);
                  setPredictedVideoUrl(null);
                  setKsyNotes([]);
                  setErrorMessage(null);
                  setIsProcessing(false);
                  setVideoLoadError(null);
                  setOutputCodec(null);
                  setInputHasAudioTrack(null);
                  setHasAudioTrack(null);
                  setTouchEvents([]);
                  setStrikeResults([]);
                  setAudioDecisionEvents([]);
                  setAvStrikeEvents([]);
                  setAlternatingOnOffSlots(null);
                  setCurrentVideoTime(0);
                  setVideoDurationSec(0);
                  setSelectedEventIndex(null);
                  setPredictionDebugSummary(null);
                }}
                className="glass px-6 py-3 rounded-xl text-sm font-medium hover:bg-primary/10 transition-colors duration-300 glow-primary"
              >
                Analyze Another Video
              </button>
            </div>
          )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Index;
