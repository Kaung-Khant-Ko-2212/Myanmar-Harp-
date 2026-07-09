import {
  AudioLines,
  BadgeCheck,
  Brain,
  BrainCircuit,
  CheckCircle2,
  Clock,
  Code2,
  Crosshair,
  DatabaseZap,
  FileVideo,
  FileText,
  Gauge,
  GitBranch,
  Grid3X3,
  Hand,
  HeartPulse,
  Layers,
  Mail,
  MonitorPlay,
  Music,
  Music2,
  PanelTop,
  ScanLine,
  ServerCog,
  ShieldCheck,
  Sparkles,
  Tag,
  Upload,
  Waypoints,
  Video,
  Waves,
  Zap,
} from 'lucide-react';
import { useEffect, useRef, useState, type ReactNode } from 'react';

const benefitCards = [
  {
    icon: MonitorPlay,
    title: 'Easy to review',
    text: 'Upload a performance, then inspect the annotated video, touch events, and predicted notes from one workspace.',
  },
  {
    icon: BrainCircuit,
    title: 'AI plus music rules',
    text: 'Vision, hand tracking, audio timing, and rule-based checks work together so each note decision is easier to understand.',
  },
  {
    icon: ShieldCheck,
    title: 'Traceable results',
    text: 'The app keeps JSON files, debug details, codec status, and cached runs available for testing and research review.',
  },
];

const techTags = ['React + Vite UI', 'FastAPI backend', 'YOLO pose model', 'MediaPipe hands', 'OpenCV video', 'Audio fusion'];

const projectMapNodes = [
  {
    icon: FileVideo,
    step: '01',
    label: 'Upload clip',
    userText: 'Choose a short saung-gauk performance video.',
    systemText: 'The backend stores the file and prepares it for analysis.',
    color: 'border-cyan-300/30 bg-cyan-300/10 text-cyan-100',
  },
  {
    icon: ScanLine,
    step: '02',
    label: 'Find strings',
    userText: 'The harp lines become visible as detected geometry.',
    systemText: 'YOLO pose and OpenCV estimate string positions frame by frame.',
    color: 'border-emerald-300/30 bg-emerald-300/10 text-emerald-100',
  },
  {
    icon: Hand,
    step: '03',
    label: 'Track fingers',
    userText: 'Touch candidates show where the player contacts strings.',
    systemText: 'MediaPipe hand tracking maps fingertips to nearby strings.',
    color: 'border-amber-300/30 bg-amber-300/10 text-amber-100',
  },
  {
    icon: AudioLines,
    step: '04',
    label: 'Check sound',
    userText: 'Audio timing helps confirm likely played notes.',
    systemText: 'Onset and motion signals are aligned with touch timing.',
    color: 'border-fuchsia-300/30 bg-fuchsia-300/10 text-fuchsia-100',
  },
  {
    icon: Music,
    step: '05',
    label: 'Export notes',
    userText: 'Review the final note events and downloadable outputs.',
    systemText: 'Fusion rules produce annotated video and event JSON files.',
    color: 'border-sky-300/30 bg-sky-300/10 text-sky-100',
  },
];

const techStack = [
  {
    icon: Code2,
    label: 'Interface',
    value: 'React + Vite',
    tag: 'Frontend',
    detail: 'Fast upload, review, and workflow screens.',
    accent: 'border-cyan-300/25 bg-cyan-300/10 text-cyan-100',
    bar: 'from-cyan-300 to-sky-300',
  },
  {
    icon: ServerCog,
    label: 'API layer',
    value: 'FastAPI',
    tag: 'Backend',
    detail: 'Receives videos and serves prediction results.',
    accent: 'border-emerald-300/25 bg-emerald-300/10 text-emerald-100',
    bar: 'from-emerald-300 to-teal-300',
  },
  {
    icon: ScanLine,
    label: 'Vision',
    value: 'YOLO + OpenCV',
    tag: 'Model',
    detail: 'Detects strings and motion from video frames.',
    accent: 'border-violet-300/25 bg-violet-300/10 text-violet-100',
    bar: 'from-violet-300 to-cyan-300',
  },
  {
    icon: Hand,
    label: 'Hands',
    value: 'MediaPipe',
    tag: 'Tracking',
    detail: 'Tracks fingertips near the detected strings.',
    accent: 'border-amber-300/25 bg-amber-300/10 text-amber-100',
    bar: 'from-amber-300 to-orange-300',
  },
  {
    icon: AudioLines,
    label: 'Audio',
    value: 'librosa + rules',
    tag: 'Signal',
    detail: 'Finds timing clues and supports note fusion.',
    accent: 'border-fuchsia-300/25 bg-fuchsia-300/10 text-fuchsia-100',
    bar: 'from-fuchsia-300 to-sky-300',
  },
  {
    icon: DatabaseZap,
    label: 'Outputs',
    value: 'JSON + cache',
    tag: 'Data',
    detail: 'Keeps event files, debug data, and repeated runs traceable.',
    accent: 'border-sky-300/25 bg-sky-300/10 text-sky-100',
    bar: 'from-sky-300 to-cyan-300',
  },
];

const modelSignals = [
  {
    label: 'String geometry',
    role: 'Where are the strings?',
    width: '92%',
    detail: 'The vision model provides the map that every touch is compared against.',
    color: 'from-cyan-300 to-emerald-300',
  },
  {
    label: 'Finger contact',
    role: 'Which string was touched?',
    width: '78%',
    detail: 'Hand landmarks turn fingertip movement into timestamped touch candidates.',
    color: 'from-amber-300 to-orange-300',
  },
  {
    label: 'Audio timing',
    role: 'When did the note sound?',
    width: '70%',
    detail: 'Audio onsets help confirm whether visual movement produced a note.',
    color: 'from-fuchsia-300 to-sky-300',
  },
  {
    label: 'Fusion decision',
    role: 'What should be exported?',
    width: '86%',
    detail: 'Rules combine the signals into final note events and debug files.',
    color: 'from-sky-300 to-cyan-300',
  },
];

const architectureStages = [
  {
    icon: Code2,
    title: 'Frontend',
    tone: 'border-blue-300/30 bg-blue-300/10 text-blue-100',
    items: ['Upload UI sends video', 'Results UI receives JSON and video URLs', 'Notes Display renders inline events'],
  },
  {
    icon: ServerCog,
    title: 'Backend API',
    tone: 'border-emerald-300/30 bg-emerald-300/10 text-emerald-100',
    items: ['POST /predict', 'GET /health', 'Orchestrator starts the pipeline'],
  },
  {
    icon: BrainCircuit,
    title: 'ML Pipeline',
    tone: 'border-fuchsia-300/30 bg-fuchsia-300/10 text-fuchsia-100',
    items: ['Video touch extraction', 'Video and audio strike inference', 'AV fusion and beat alignment'],
  },
  {
    icon: DatabaseZap,
    title: 'Data Stores',
    tone: 'border-amber-300/30 bg-amber-300/10 text-amber-100',
    items: ['Uploaded and annotated videos', 'Touch event JSON', 'Strike event and slot JSON'],
  },
];

const teamMembers = [
  {
    name: 'Hsu Sandy Hnin',
    role: 'Labeling + Model Training',
    focus: 'Prepared labeled training data and supported model training for detection quality.',
    icon: Brain,
    accent: 'border-fuchsia-300/30 bg-fuchsia-300/10 text-fuchsia-100',
    glow: 'from-fuchsia-300/35 to-cyan-300/10',
  },
  {
    name: 'Phyu Sin Thant',
    role: 'Labeling + Post Modeling + Video Processing',
    focus: 'Handled labeling support, post-modeling review, and video processing workflow.',
    icon: Video,
    accent: 'border-sky-300/30 bg-sky-300/10 text-sky-100',
    glow: 'from-sky-300/35 to-emerald-300/10',
  },
  {
    name: 'Kaung Khant Ko',
    role: 'UI + Labeling + Vibration + Audio + Raw JSON',
    focus: 'Built interface pieces and supported vibration, audio, labeling, and raw JSON outputs.',
    icon: AudioLines,
    accent: 'border-amber-300/30 bg-amber-300/10 text-amber-100',
    glow: 'from-amber-300/35 to-fuchsia-300/10',
  },
  {
    name: 'Htet Aung Shine',
    role: 'Hand Detection + Touch',
    focus: 'Worked on hand detection and touch-event mapping for string interaction analysis.',
    icon: Hand,
    accent: 'border-emerald-300/30 bg-emerald-300/10 text-emerald-100',
    glow: 'from-emerald-300/35 to-cyan-300/10',
  },
  {
    name: 'Kyaw Htet Win',
    role: 'Labeling + Notes Output',
    focus: 'Supported labeling work and note-output review for the final analysis results.',
    icon: Music2,
    accent: 'border-cyan-300/30 bg-cyan-300/10 text-cyan-100',
    glow: 'from-cyan-300/35 to-violet-300/10',
  },
];

const ClockIcon = () => <Clock className="h-5 w-5" />;
const UploadIcon = () => <Upload className="h-5 w-5" />;
const HeartIcon = () => <HeartPulse className="h-5 w-5" />;
const VideoIcon = () => <Video className="h-5 w-5" />;
const CrosshairIcon = () => <Crosshair className="h-5 w-5" />;
const GridIcon = () => <Grid3X3 className="h-5 w-5" />;
const FlowIcon = () => <GitBranch className="h-5 w-5" />;
const ZapIcon = () => <Zap className="h-5 w-5" />;
const WaveIcon = () => <Waves className="h-5 w-5" />;
const OverlayIcon = () => <PanelTop className="h-5 w-5" />;
const FileIcon = () => <FileText className="h-5 w-5" />;

type ArchitecturePanelProps = {
  title: string;
  label?: string;
  className?: string;
  children: ReactNode;
};

const ArchitecturePanel = ({ title, label, className = '', children }: ArchitecturePanelProps) => (
  <div className={`absolute z-30 rounded-2xl border bg-slate-950/70 p-2 shadow-[0_18px_44px_rgba(0,0,0,0.28)] backdrop-blur-md ${className}`}>
    <div className="mb-2 inline-flex rounded-lg border border-white/15 bg-white/10 px-2 py-1 text-[10px] font-bold uppercase tracking-[0.04em] text-cyan-100">
      {label || title}
    </div>
    {children}
  </div>
);

type ArchitectureLineProps = {
  className: string;
  dashed?: boolean;
  arrow?: boolean;
  label?: string;
  labelClassName?: string;
};

const ArchitectureLine = ({ className, dashed = false, arrow = true, label, labelClassName = '' }: ArchitectureLineProps) => (
  <div className={`absolute z-20 ${className}`}>
    <div
      className={`relative h-full w-full ${
        dashed ? 'border-t-2 border-dashed border-cyan-200/45' : 'border-t-2 border-cyan-100/75'
      }`}
    >
      {arrow ? (
        <span className="absolute -right-1.5 -top-[6px] h-0 w-0 border-y-[6px] border-l-[10px] border-y-transparent border-l-cyan-100 drop-shadow-[0_0_8px_rgba(103,232,249,0.55)]" />
      ) : null}
      {label ? (
        <span className={`absolute -top-6 whitespace-nowrap rounded-full border border-cyan-200/20 bg-slate-950/80 px-2 py-0.5 text-[10px] font-semibold text-cyan-100 shadow-sm ${labelClassName}`}>
          {label}
        </span>
      ) : null}
    </div>
  </div>
);

const ArchitectureVerticalLine = ({ className, dashed = false }: { className: string; dashed?: boolean }) => (
  <div className={`absolute z-20 ${className} ${dashed ? 'border-l-2 border-dashed border-cyan-200/45' : 'border-l-2 border-cyan-100/75'}`} />
);

const MiniStep = ({ icon, label }: { icon: ReactNode; label: string }) => (
  <div className="flex min-w-0 flex-col items-center gap-1 text-center">
    <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-cyan-200/35 bg-cyan-200/10 text-cyan-100 shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]">
      {icon}
    </div>
    <span className="max-w-[76px] text-[9px] font-bold leading-tight text-white/90">{label}</span>
  </div>
);

const ARCHITECTURE_WIDTH = 1680;
const ARCHITECTURE_HEIGHT = 520;

const LegacySystemArchitectureDiagram = () => {
  const diagramFrameRef = useRef<HTMLDivElement>(null);
  const [diagramScale, setDiagramScale] = useState(1);

  useEffect(() => {
    const updateScale = () => {
      const width = diagramFrameRef.current?.clientWidth ?? ARCHITECTURE_WIDTH;
      setDiagramScale(Math.min(1, width / ARCHITECTURE_WIDTH));
    };

    updateScale();

    if (typeof ResizeObserver === 'undefined' || !diagramFrameRef.current) {
      window.addEventListener('resize', updateScale);
      return () => window.removeEventListener('resize', updateScale);
    }

    const observer = new ResizeObserver(updateScale);
    observer.observe(diagramFrameRef.current);

    return () => observer.disconnect();
  }, []);

  return (
    <div className="mt-14">
      <div className="mb-7 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-fuchsia-300/25 bg-fuchsia-300/10">
              <BrainCircuit className="h-5 w-5 text-fuchsia-200" />
            </div>
            <p className="text-sm font-medium uppercase tracking-[0.22em] text-cyan-300/80">System Architecture</p>
          </div>
          <h3 className="mt-3 font-heading text-2xl font-bold text-white md:text-4xl">
            How the frontend, API, models, and outputs connect.
          </h3>
        </div>
        <p className="max-w-xl text-sm leading-6 text-muted-foreground">
          The architecture separates upload, orchestration, ML inference, fusion, annotation, and saved artifacts.
        </p>
      </div>

      <div ref={diagramFrameRef} className="hidden overflow-hidden rounded-2xl border border-white/10 bg-white/[0.025] p-3 shadow-[0_24px_80px_rgba(0,0,0,0.28)] backdrop-blur-xl lg:block">
        <div
          className="relative w-full overflow-hidden"
          style={{ height: `${ARCHITECTURE_HEIGHT * diagramScale}px` }}
        >
          <div
            className="relative h-[520px] w-[1680px] origin-top-left overflow-hidden rounded-xl bg-transparent text-white"
            style={{ transform: `scale(${diagramScale})` }}
          >
          <ArchitecturePanel title="Config" label="CONFIG" className="left-[10px] top-[60px] h-[110px] w-[140px] border-slate-300/35">
            <div className="grid grid-cols-2 gap-3 pt-2">
              <MiniStep icon={<Gauge className="h-5 w-5" />} label="Fast Mode" />
              <MiniStep icon={<ClockIcon />} label="Timing Profile" />
            </div>
          </ArchitecturePanel>

          <ArchitecturePanel title="Frontend" label="FRONTEND" className="left-[10px] top-[230px] h-[235px] w-[140px] border-blue-300/55 bg-blue-300/10">
            <div className="space-y-3 pt-1">
              <MiniStep icon={<UploadIcon />} label="Upload UI" />
              <MiniStep icon={<MonitorPlay className="h-5 w-5" />} label="Results UI" />
              <MiniStep icon={<Music className="h-5 w-5" />} label="Notes Display" />
            </div>
          </ArchitecturePanel>

          <ArchitecturePanel title="Backend API" label="BACKEND API" className="left-[275px] top-[75px] h-[210px] w-[250px] border-emerald-300/55 bg-emerald-300/10">
            <div className="absolute left-4 top-10 h-[150px] w-[145px] rounded-xl border border-blue-300/45 bg-blue-300/10 p-2">
              <div className="mb-3 inline-flex rounded-lg border border-blue-200/25 bg-blue-200/10 px-1.5 py-0.5 text-[10px] font-bold text-blue-100">
                API ENDPOINTS
              </div>
              <div className="grid grid-cols-2 gap-2">
                <MiniStep icon={<FileVideo className="h-5 w-5" />} label="POST /predict" />
                <MiniStep icon={<HeartIcon />} label="GET /health" />
              </div>
              <div className="mt-4 flex justify-center">
                <MiniStep icon={<VideoIcon />} label="POST /api/predict-video" />
              </div>
            </div>
            <div className="absolute right-5 top-[122px] flex flex-col items-center">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-emerald-200/35 bg-emerald-200/10 text-emerald-100">
                <ServerCog className="h-5 w-5" />
              </div>
              <span className="mt-2 text-[10px] font-bold text-white/90">Orchestrator</span>
            </div>
          </ArchitecturePanel>

          <ArchitecturePanel title="ML Pipeline" label="ML PIPELINE" className="left-[555px] top-[30px] h-[410px] w-[835px] border-fuchsia-300/60 bg-fuchsia-300/10">
            <div className="absolute left-4 top-[115px] h-[150px] w-[270px] rounded-xl border border-blue-300/45 bg-blue-300/10 p-2">
              <div className="mb-6 inline-flex rounded-lg border border-blue-200/25 bg-blue-200/10 px-1.5 py-0.5 text-[10px] font-bold text-blue-100">
                VIDEO TOUCH EXTRACTION
              </div>
              <div className="grid grid-cols-4 gap-2">
                <MiniStep icon={<ScanLine className="h-5 w-5" />} label="YOLO Harp Detection" />
                <MiniStep icon={<Hand className="h-5 w-5" />} label="MediaPipe Tracking" />
                <MiniStep icon={<CrosshairIcon />} label="Touch Detection" />
                <MiniStep icon={<GridIcon />} label="Video Writer" />
              </div>
            </div>

            <div className="absolute left-[320px] top-[155px] h-[100px] w-[230px] rounded-xl border border-blue-300/45 bg-blue-300/10 p-2">
              <div className="mb-4 inline-flex rounded-lg border border-blue-200/25 bg-blue-200/10 px-1.5 py-0.5 text-[10px] font-bold text-blue-100">
                VIDEO STRIKE INFERENCE
              </div>
              <div className="grid grid-cols-2 gap-6">
                <MiniStep icon={<FlowIcon />} label="Optical Flow" />
                <MiniStep icon={<CheckCircle2 className="h-5 w-5" />} label="Rule Engine" />
              </div>
            </div>

            <div className="absolute left-[320px] top-[285px] h-[110px] w-[230px] rounded-xl border border-blue-300/45 bg-blue-300/10 p-2">
              <div className="mb-4 inline-flex rounded-lg border border-blue-200/25 bg-blue-200/10 px-1.5 py-0.5 text-[10px] font-bold text-blue-100">
                AUDIO STRIKE INFERENCE
              </div>
              <div className="grid grid-cols-3 gap-2">
                <MiniStep icon={<AudioLines className="h-5 w-5" />} label="Audio Extraction" />
                <MiniStep icon={<ZapIcon />} label="Onset Detection" />
                <MiniStep icon={<WaveIcon />} label="Pitch Matching" />
              </div>
            </div>

            <div className="absolute left-[575px] top-[244px] h-[90px] w-[92px] rounded-xl border border-blue-300/45 bg-blue-300/10 p-2">
              <div className="mb-2 inline-flex rounded-lg border border-blue-200/25 bg-blue-200/10 px-1.5 py-0.5 text-[10px] font-bold text-blue-100">
                AV FUSION
              </div>
              <MiniStep icon={<Layers className="h-5 w-5" />} label="Fuse Decisions" />
            </div>

            <div className="absolute left-[650px] top-[40px] h-[110px] w-[143px] rounded-xl border border-blue-300/45 bg-blue-300/10 p-2">
              <div className="mb-3 inline-flex rounded-lg border border-blue-200/25 bg-blue-200/10 px-1.5 py-0.5 text-[10px] font-bold text-blue-100">
                AV ANNOTATION
              </div>
              <div className="grid grid-cols-2 gap-3">
                <MiniStep icon={<OverlayIcon />} label="Overlay Renderer" />
                <MiniStep icon={<GridIcon />} label="H264 Transcode" />
              </div>
            </div>

            <div className="absolute left-[650px] top-[285px] h-[80px] w-[143px] rounded-xl border border-blue-300/45 bg-blue-300/10 p-2">
              <div className="mb-3 inline-flex rounded-lg border border-blue-200/25 bg-blue-200/10 px-1.5 py-0.5 text-[10px] font-bold text-blue-100">
                BEAT ALIGNMENT
              </div>
              <div className="grid grid-cols-2 gap-3">
                <MiniStep icon={<WaveIcon />} label="Librosa Beats" />
                <MiniStep icon={<GridIcon />} label="Slot Summary" />
              </div>
            </div>
          </ArchitecturePanel>

          <ArchitecturePanel title="Data Stores" label="DATA STORES" className="left-[1435px] top-[13px] h-[494px] w-[125px] border-orange-300/65 bg-orange-300/10">
            <div className="rounded-xl border border-blue-300/45 bg-blue-300/10 p-2">
              <div className="mb-3 inline-flex rounded-lg border border-blue-200/25 bg-blue-200/10 px-1.5 py-0.5 text-[10px] font-bold text-blue-100">
                VIDEO FILES
              </div>
              <MiniStep icon={<UploadIcon />} label="uploads/<id>.mp4" />
              <div className="mt-4">
                <MiniStep icon={<GridIcon />} label="*_annotated_av.mp4" />
              </div>
            </div>
            <div className="mt-5 rounded-xl border border-blue-300/45 bg-blue-300/10 p-2">
              <div className="mb-3 inline-flex rounded-lg border border-blue-200/25 bg-blue-200/10 px-1.5 py-0.5 text-[10px] font-bold text-blue-100">
                JSON ARTIFACTS
              </div>
              <div className="space-y-4">
                <MiniStep icon={<FileIcon />} label="touch_events.json" />
                <MiniStep icon={<FileIcon />} label="av_strike_events.json" />
                <MiniStep icon={<FileIcon />} label="on_off_slots.json" />
              </div>
            </div>
          </ArchitecturePanel>

          <div className="absolute left-[1570px] top-[13px] w-[105px] rounded-xl border border-white/10 bg-slate-950/70 p-2 text-[9px] font-bold text-white/85 shadow-sm backdrop-blur-md">
            <div className="mb-3 flex items-center gap-2">
              <span className="h-0.5 w-7 bg-cyan-100/75" />
              <span>Main data flow</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="h-px w-7 border-t-2 border-dashed border-cyan-200/45" />
              <span>Async/inline response</span>
            </div>
          </div>

          <ArchitectureLine className="left-[150px] top-[115px] w-[305px]" dashed arrow={false} label="tuning params" />
          <ArchitectureVerticalLine className="left-[455px] top-[115px] h-[82px]" dashed />
          <ArchitectureLine className="left-[455px] top-[197px] w-[42px]" dashed />
          <ArchitectureLine className="left-[112px] top-[282px] w-[180px]" arrow={false} label="video upload" labelClassName="left-[88px]" />
          <ArchitectureVerticalLine className="left-[170px] top-[213px] h-[69px]" />
          <ArchitectureLine className="left-[170px] top-[213px] w-[122px]" />
          <ArchitectureLine className="left-[505px] top-[217px] w-[65px]" />
          <ArchitectureLine className="left-[626px] top-[216px] w-[26px]" />
          <ArchitectureLine className="left-[690px] top-[216px] w-[22px]" arrow={false} />
          <ArchitectureLine className="left-[736px] top-[216px] w-[36px]" arrow={false} />
          <ArchitectureLine className="left-[796px] top-[216px] w-[69px]" />
          <ArchitectureLine className="left-[796px] top-[340px] w-[69px]" />
          <ArchitectureVerticalLine className="left-[840px] top-[216px] h-[124px]" />
          <ArchitectureLine className="left-[1050px] top-[230px] w-[40px]" />
          <ArchitectureLine className="left-[1050px] top-[335px] w-[40px]" />
          <ArchitectureLine className="left-[667px] top-[292px] w-[768px]" />
          <ArchitectureVerticalLine className="left-[1185px] top-[125px] h-[167px]" />
          <ArchitectureLine className="left-[1185px] top-[125px] w-[20px]" arrow={false} />
          <ArchitectureLine className="left-[1348px] top-[125px] w-[102px]" />
          <ArchitectureLine className="left-[1348px] top-[364px] w-[102px]" />
          <ArchitectureLine className="left-[1450px] top-[352px] w-[1px]" />
          <ArchitectureLine className="left-[170px] top-[455px] w-[1280px]" dashed arrow={false} label="inline events" labelClassName="left-[230px]" />
          <ArchitectureVerticalLine className="left-[170px] top-[350px] h-[105px]" dashed />
          <ArchitectureLine className="left-[92px] top-[350px] w-[78px]" dashed />
          <ArchitectureLine className="left-[170px] top-[485px] w-[1280px]" dashed arrow={false} label="slot data" labelClassName="left-[230px]" />
          <ArchitectureVerticalLine className="left-[170px] top-[397px] h-[88px]" dashed />
          <ArchitectureLine className="left-[92px] top-[397px] w-[78px]" dashed />
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:hidden">
        {architectureStages.map((stage, index) => {
          const Icon = stage.icon;
          return (
            <article key={stage.title} className="rounded-2xl border border-white/10 bg-card/70 p-5 backdrop-blur-xl">
              <div className="mb-4 flex items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <div className={`flex h-12 w-12 items-center justify-center rounded-xl border ${stage.tone}`}>
                    <Icon className="h-5 w-5" />
                  </div>
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.18em] text-white/45">
                      {String(index + 1).padStart(2, '0')}
                    </p>
                    <h4 className="text-base font-semibold text-white">{stage.title}</h4>
                  </div>
                </div>
              </div>
              <div className="space-y-3">
                {stage.items.map((item) => (
                  <div key={item} className="flex gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-3">
                    <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-cyan-200" />
                    <p className="text-sm leading-5 text-muted-foreground">{item}</p>
                  </div>
                ))}
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
};
const SystemArchitectureDiagram = () => {
  return (
    <div className="mt-14">
      <div className="mb-7 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-fuchsia-300/25 bg-fuchsia-300/10">
              <BrainCircuit className="h-5 w-5 text-fuchsia-200" />
            </div>
            <p className="text-sm font-medium uppercase tracking-[0.22em] text-cyan-300/80">System Architecture</p>
          </div>
          <h3 className="mt-3 font-heading text-2xl font-bold text-white md:text-4xl">
            How the frontend, API, models, and outputs connect.
          </h3>
        </div>
        <p className="max-w-xl text-sm leading-6 text-muted-foreground">
          The architecture separates upload, orchestration, ML inference, fusion, annotation, and saved artifacts.
        </p>
      </div>

      <figure className="architecture-figure overflow-hidden rounded-2xl border border-white/10 bg-transparent p-4 shadow-[0_24px_80px_rgba(0,0,0,0.18)] backdrop-blur-sm">
        <div className="relative flex justify-center">
          <img
            src="/system-architecture-transparent.png"
            alt="System architecture design showing the frontend, backend API, video and audio inference, fusion decisions, beat alignment, data stores, and video stores."
            width="3200"
            height="2125"
            loading="lazy"
            className="architecture-image block h-auto max-w-full object-contain"
            style={{ maxHeight: 'min(60vh, 640px)' }}
          />
        </div>
      </figure>

      <div className="mt-4 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {architectureStages.map((stage, index) => {
          const Icon = stage.icon;
          return (
            <article
              key={stage.title}
              className="group lift-card animate-card-enter rounded-2xl border border-white/10 bg-card/70 p-5 backdrop-blur-xl hover:border-cyan-300/25 hover:bg-card/80"
              style={{ animationDelay: `${index * 70}ms` }}
            >
              <div className="mb-4 flex items-center gap-3">
                <div className={`icon-tile flex h-12 w-12 items-center justify-center rounded-xl border ${stage.tone}`}>
                  <Icon className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-white/45">
                    {String(index + 1).padStart(2, '0')}
                  </p>
                  <h4 className="text-base font-semibold text-white">{stage.title}</h4>
                </div>
              </div>
              <div className="space-y-3">
                {stage.items.map((item) => (
                  <div key={item} className="flex gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-3">
                    <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-cyan-200" />
                    <p className="text-sm leading-5 text-muted-foreground">{item}</p>
                  </div>
                ))}
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
};

const SiteSections = () => {
  return (
    <div className="relative z-10">
      <section id="workflow" className="px-4 pb-20 pt-12 md:pt-14">
        <div className="mx-auto max-w-7xl">
          <div className="mx-auto max-w-4xl text-center">
            <p className="text-sm font-medium uppercase tracking-[0.22em] text-cyan-300/80">Workflow</p>
            <h2 className="mt-3 font-heading text-3xl font-bold leading-tight text-white md:text-5xl">
              See how a harp performance becomes clean note data.
            </h2>
            <p className="mx-auto mt-5 max-w-3xl text-sm leading-7 text-muted-foreground md:text-base">
              The project combines a friendly React interface with a FastAPI analysis backend. It uses vision,
              hand tracking, audio timing, and clear decision rules so users can review every step instead of
              only seeing a final answer.
            </p>
          </div>

          <div className="mx-auto mt-6 flex max-w-4xl flex-wrap justify-center gap-2">
            {techTags.map((tag) => (
              <span
                key={tag}
                className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-white/75"
              >
                {tag}
              </span>
            ))}
          </div>

          <div className="mt-10 grid gap-4 md:grid-cols-3">
            {benefitCards.map((item) => {
              const Icon = item.icon;
              return (
                <article
                  key={item.title}
                  className="group lift-card animate-card-enter rounded-2xl border border-white/10 bg-card/60 p-5 backdrop-blur-xl hover:border-cyan-300/25 hover:bg-card/75"
                >
                  <div className="icon-tile mb-4 flex h-12 w-12 items-center justify-center rounded-xl border border-cyan-300/25 bg-cyan-300/10">
                    <Icon className="h-5 w-5 text-cyan-200" />
                  </div>
                  <h3 className="text-base font-semibold text-white">{item.title}</h3>
                  <p className="mt-2 text-sm leading-6 text-muted-foreground">{item.text}</p>
                </article>
              );
            })}
          </div>

          <SystemArchitectureDiagram />

          <div className="mt-14">
            <div className="mb-7 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
              <div>
                <div className="flex items-center gap-3">
                  <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-cyan-300/25 bg-cyan-300/10">
                    <Waypoints className="h-5 w-5 text-cyan-200" />
                  </div>
                  <p className="text-sm font-medium uppercase tracking-[0.22em] text-cyan-300/80">Project Map</p>
                </div>
                <h3 className="mt-3 font-heading text-2xl font-bold text-white md:text-4xl">
                  A simple path from clip to notes.
                </h3>
              </div>
              <p className="max-w-xl text-sm leading-6 text-muted-foreground">
                Each stage shows what the user sees and what the system checks behind the scenes.
              </p>
            </div>

            <div className="relative">
              <div className="absolute left-10 right-10 top-10 hidden h-1 rounded-full bg-white/10 lg:block" />
              <div className="absolute left-10 right-10 top-10 hidden h-1 rounded-full bg-gradient-to-r from-cyan-300 via-amber-300 to-sky-300 opacity-50 lg:block" />
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
                {projectMapNodes.map((node) => {
                  const Icon = node.icon;
                  return (
                    <article
                      key={node.label}
                      className="group lift-card animate-card-enter relative rounded-2xl border border-white/10 bg-card/70 p-5 backdrop-blur-xl hover:border-cyan-300/20 hover:bg-card/80"
                      style={{ animationDelay: `${Number(node.step) * 55}ms` }}
                    >
                      <div className="mb-5 flex items-center justify-between gap-3">
                        <div className={`icon-tile flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl border ${node.color}`}>
                          <Icon className="h-6 w-6" />
                        </div>
                        <span className="font-heading text-2xl font-bold text-white/25">{node.step}</span>
                      </div>
                      <h4 className="text-base font-semibold text-white">{node.label}</h4>
                      <p className="mt-2 text-sm leading-6 text-white/70">{node.userText}</p>
                      <div className="mt-4 flex gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-3">
                        <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-cyan-200" />
                        <p className="text-xs leading-5 text-muted-foreground">{node.systemText}</p>
                      </div>
                    </article>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="mt-20 grid gap-5 lg:grid-cols-[1fr_1fr]">
            <article className="h-full rounded-2xl border border-white/10 bg-[linear-gradient(135deg,rgba(13,24,28,0.78),rgba(10,12,19,0.94))] p-5 backdrop-blur-xl md:p-6">
              <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div className="flex items-center gap-3">
                  <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-emerald-300/25 bg-emerald-300/10">
                    <Layers className="h-6 w-6 text-emerald-200" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-white">Technology Stack</h3>
                    <p className="mt-1 text-sm leading-6 text-muted-foreground">
                      The main tools are grouped by what they do in the workflow.
                    </p>
                  </div>
                </div>
                <span className="w-fit rounded-full border border-emerald-300/20 bg-emerald-300/10 px-3 py-1 text-xs font-medium text-emerald-100">
                  6 layers
                </span>
              </div>

              <div className="grid auto-rows-fr gap-3 sm:grid-cols-2">
                {techStack.map((item) => {
                  const Icon = item.icon;
                  return (
                    <div
                      key={item.label}
                      className="group lift-card relative flex min-h-[220px] flex-col overflow-hidden rounded-2xl border border-white/10 bg-white/[0.035] p-4 hover:border-white/20 hover:bg-white/[0.055]"
                    >
                      <div className={`absolute inset-x-0 top-0 h-1 bg-gradient-to-r ${item.bar} transition-opacity duration-300 group-hover:opacity-90`} />
                      <div className="mb-4 flex items-start justify-between gap-3">
                        <div className={`icon-tile flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border ${item.accent}`}>
                          <Icon className="h-5 w-5" />
                        </div>
                        <span className="rounded-full border border-white/10 bg-black/20 px-2.5 py-1 text-[11px] font-medium text-white/55">
                          {item.tag}
                        </span>
                      </div>
                      <div>
                        <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-white/45">{item.label}</p>
                        <h4 className="mt-1 text-base font-semibold text-white">{item.value}</h4>
                        <p className="mt-3 text-sm leading-6 text-muted-foreground">{item.detail}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </article>

            <article className="flex h-full flex-col rounded-2xl border border-white/10 bg-[linear-gradient(135deg,rgba(15,23,42,0.82),rgba(8,13,22,0.95))] p-5 backdrop-blur-xl md:p-6">
              <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div className="flex items-center gap-3">
                  <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-cyan-300/25 bg-cyan-300/10">
                    <Gauge className="h-5 w-5 text-cyan-200" />
                  </div>
                  <div>
                    <h3 className="text-base font-semibold text-white">Model Signal Graph</h3>
                    <p className="text-sm text-muted-foreground">How each model contributes to the final decision.</p>
                  </div>
                </div>
                <span className="w-fit rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-white/70">
                  Vision + hands + audio
                </span>
              </div>

              <div className="mb-4 grid grid-cols-4 text-[10px] font-medium uppercase tracking-[0.14em] text-white/35">
                <span>Input</span>
                <span>Detect</span>
                <span>Confirm</span>
                <span className="text-right">Export</span>
              </div>

              <div className="grid flex-1 grid-rows-4 gap-4">
                {modelSignals.map((signal) => (
                  <div
                    key={signal.label}
                    className="group lift-card flex flex-col justify-center rounded-2xl border border-white/10 bg-white/[0.025] p-4 hover:border-cyan-300/20 hover:bg-white/[0.04]"
                  >
                    <div className="mb-2 flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
                      <div>
                        <p className="text-sm font-semibold text-white">{signal.label}</p>
                        <p className="text-xs text-cyan-100/70">{signal.role}</p>
                      </div>
                    </div>
                    <div className="h-3 overflow-hidden rounded-full bg-white/10">
                      <div
                        className={`h-full rounded-full bg-gradient-to-r ${signal.color} transition-[width,filter] duration-500 group-hover:brightness-125`}
                        style={{ width: signal.width }}
                      />
                    </div>
                    <p className="mt-2 text-xs leading-5 text-muted-foreground">{signal.detail}</p>
                  </div>
                ))}
              </div>
            </article>
          </div>

        </div>
      </section>

      <section id="about" className="px-4 pb-20 pt-12 md:pt-14">
        <div className="mx-auto max-w-7xl">
          <div>
            <div className="mx-auto mb-10 max-w-4xl text-center">
              <p className="text-sm font-medium uppercase tracking-[0.22em] text-cyan-300/80">About</p>
              <h3 className="mt-3 font-heading text-3xl font-bold leading-tight text-white md:text-5xl">
                About the Team
              </h3>
              <p className="mx-auto mt-5 max-w-3xl text-sm leading-7 text-muted-foreground md:text-base">
                Five contributors support the analysis workflow across vision, audio, interface design, and validation.
              </p>
            </div>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
              {teamMembers.map((member, index) => {
                const Icon = member.icon;
                return (
                  <article
                    key={member.name}
                    className="group lift-card animate-card-enter relative min-h-[260px] overflow-hidden rounded-2xl border border-white/10 bg-card/60 p-5 backdrop-blur-xl hover:border-white/20 hover:bg-card/75"
                    style={{ animationDelay: `${index * 90}ms` }}
                  >
                    <div className={`absolute -right-10 -top-10 h-28 w-28 rounded-full bg-gradient-to-br ${member.glow} blur-2xl transition-opacity duration-300 group-hover:opacity-90`} />
                    <div className="absolute right-4 top-4 flex h-7 w-7 items-center justify-center rounded-full border border-white/10 bg-white/5 text-cyan-100 transition-transform duration-500 group-hover:rotate-12 group-hover:scale-110">
                      <Sparkles className="h-3.5 w-3.5" />
                    </div>

                    <div className="relative">
                      <div className="mb-5 flex items-center gap-3">
                        <div className={`icon-tile flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl border ${member.accent}`}>
                          <Icon className="h-6 w-6" />
                        </div>
                        <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-white/[0.04] font-heading text-sm font-bold text-white/60">
                          {String(index + 1).padStart(2, '0')}
                        </div>
                      </div>

                      <h4 className="text-base font-semibold leading-snug text-white">{member.name}</h4>
                      <div className="mt-3 inline-flex items-center gap-1.5 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-[11px] font-medium text-cyan-100">
                        <BadgeCheck className="h-3.5 w-3.5" />
                        {member.role}
                      </div>
                      <p className="mt-4 text-sm leading-6 text-muted-foreground">{member.focus}</p>

                      <div className="mt-5 flex items-center gap-2 text-xs font-medium text-white/45">
                        <Tag className="h-3.5 w-3.5 text-cyan-200/70" />
                        Project contributor
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      <section id="contact" className="px-4 pb-20 pt-12 md:pt-14">
        <div className="mx-auto max-w-5xl overflow-hidden rounded-3xl border border-white/10 bg-[linear-gradient(135deg,rgba(34,211,238,0.14),rgba(168,85,247,0.14))] p-8 backdrop-blur-xl md:p-10">
          <div className="grid gap-8 md:grid-cols-[1.1fr_0.9fr] md:items-center">
            <div>
              <p className="text-sm font-medium uppercase tracking-[0.22em] text-cyan-200/85">Contact</p>
              <h2 className="mt-3 font-heading text-3xl font-bold text-white md:text-5xl">
                Need to test another performance set?
              </h2>
              <p className="mt-4 text-sm leading-7 text-white/70 md:text-base">
                Use the analyzer for local experiments, or prepare a batch of clips and notes for a deeper review of string mapping accuracy.
              </p>
            </div>

            <div className="group lift-card rounded-2xl border border-white/12 bg-black/25 p-5 hover:border-cyan-300/25 hover:bg-black/30">
              <div className="flex items-center gap-3">
                <div className="icon-tile flex h-11 w-11 items-center justify-center rounded-xl border border-cyan-300/25 bg-cyan-300/10">
                  <Mail className="h-5 w-5 text-cyan-200" />
                </div>
                <div>
                  <p className="text-sm font-medium text-white">Project contact</p>
                  <p className="text-xs text-white/60">Research and analysis workflow</p>
                </div>
              </div>
              <a
                href="mailto:research@example.com?subject=Myanmar%20Harp%20Analysis"
                className="mt-5 inline-flex w-full items-center justify-center rounded-xl border border-cyan-300/35 bg-cyan-300/12 px-4 py-3 text-sm font-medium text-cyan-100 transition-colors hover:bg-cyan-300/20"
              >
                Send Message
              </a>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default SiteSections;
