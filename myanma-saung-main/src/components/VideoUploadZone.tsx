import { useState, useCallback, DragEvent, ChangeEvent } from 'react';
import { FileVideo, UploadCloud, X } from 'lucide-react';

interface VideoUploadZoneProps {
  onUpload: (file: File) => void;
  isProcessing: boolean;
}

const VideoUploadZone = ({ onUpload, isProcessing }: VideoUploadZoneProps) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleFile = useCallback((file: File) => {
    if (file.type.startsWith('video/')) {
      setFileName(file.name);
      const url = URL.createObjectURL(file);
      setPreview(url);
      onUpload(file);
    }
  }, [onUpload]);

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const clearPreview = () => {
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setFileName(null);
  };

  if (isProcessing) {
    return null;
  }

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className="h-full w-full"
    >
      <div
        className={`relative flex h-full min-h-[18rem] flex-col justify-center overflow-hidden rounded-xl border transition-colors duration-300 ${
          isDragOver ? 'border-cyan-300/70 bg-cyan-300/10' : 'border-white/10 bg-black/20'
        } ${preview ? 'p-3' : 'p-4 md:p-5'}`}
      >
        {preview ? (
          <div>
            <div className="mb-3 flex items-center justify-between gap-3">
              <div className="min-w-0">
                <p className="text-sm font-medium text-white">Selected video</p>
                <p className="truncate text-xs text-muted-foreground">{fileName}</p>
              </div>
              <button
                type="button"
                onClick={clearPreview}
                className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-white/80 transition-colors hover:bg-destructive/20"
              >
                <X className="h-3.5 w-3.5" />
                Remove
              </button>
            </div>
            <video
              src={preview}
              className="aspect-video w-full rounded-lg border border-white/10 bg-black/40 object-contain"
              controls
              muted
            />
          </div>
        ) : (
          <label
            className={`group flex h-full cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-6 text-center transition-colors ${
              isDragOver
                ? 'border-cyan-300/80 bg-cyan-300/10'
                : 'border-white/15 bg-white/5 hover:border-cyan-300/45 hover:bg-white/10'
            }`}
          >
            <input
              type="file"
              accept="video/*"
              onChange={handleInputChange}
              className="sr-only"
            />

            <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-xl border border-white/10 bg-white/10">
              {isDragOver ? (
                <FileVideo className="h-6 w-6 text-cyan-200" />
              ) : (
                <UploadCloud className="h-6 w-6 text-cyan-200" />
              )}
            </div>

            <h3 className="text-lg font-semibold text-white">
              {isDragOver ? 'Release to upload' : 'Choose your performance video'}
            </h3>

            <p className="mt-2 max-w-md text-sm leading-6 text-muted-foreground">
              Click the button below or drag a video file into this area.
            </p>

            <span className="mt-4 inline-flex h-10 items-center justify-center rounded-lg bg-cyan-300 px-5 text-sm font-semibold text-slate-950 transition-transform group-hover:-translate-y-0.5">
              Choose Video
            </span>

            <div className="mt-4 flex flex-wrap items-center justify-center gap-2 text-xs text-white/55">
              <span className="rounded-md border border-white/10 bg-black/20 px-2.5 py-1">MP4</span>
              <span className="rounded-md border border-white/10 bg-black/20 px-2.5 py-1">MOV</span>
              <span className="rounded-md border border-white/10 bg-black/20 px-2.5 py-1">WEBM</span>
            </div>
          </label>
        )}
      </div>
    </div>
  );
};

export default VideoUploadZone;
