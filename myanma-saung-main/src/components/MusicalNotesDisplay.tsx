import { useEffect, useState } from 'react';
import { Music, Music2, Music3, Music4 } from 'lucide-react';

interface DetectedString {
  id: number;
  name: string;
  frequency: string;
  confidence: number;
}

interface GeneratedNote {
  note: string;
  octave: number;
  duration: string;
  timestamp: string;
}

interface MusicalNotesDisplayProps {
  isVisible: boolean;
  alternatingSlots?: AlternatingOnOffSlotsPayload | null;
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
  slot_start?: string;
  sequence_length?: number;
  sequence?: Array<Record<string, AlternatingSlotEntryPayload>>;
}

interface AlternatingSlotCell {
  kind: 'on_beat' | 'off_beat';
  data: AlternatingSlotEntryPayload;
}

interface AlternatingPairCell {
  onBeat: AlternatingSlotEntryPayload | null;
  offBeat: AlternatingSlotEntryPayload | null;
}

const mockStrings: DetectedString[] = [
  { id: 1, name: 'String 1 (Tha)', frequency: '261.6 Hz', confidence: 98 },
  { id: 2, name: 'String 2 (Kya)', frequency: '293.7 Hz', confidence: 95 },
  { id: 3, name: 'String 3 (Gyi)', frequency: '329.6 Hz', confidence: 97 },
  { id: 4, name: 'String 4 (Gha)', frequency: '349.2 Hz', confidence: 92 },
  { id: 5, name: 'String 5 (Nge)', frequency: '392.0 Hz', confidence: 96 },
];

const mockNotes: GeneratedNote[] = [
  { note: 'C', octave: 4, duration: '♩', timestamp: '0:00' },
  { note: 'E', octave: 4, duration: '♪', timestamp: '0:02' },
  { note: 'G', octave: 4, duration: '♩', timestamp: '0:04' },
  { note: 'A', octave: 4, duration: '♫', timestamp: '0:06' },
  { note: 'G', octave: 4, duration: '♩', timestamp: '0:08' },
  { note: 'E', octave: 4, duration: '♪', timestamp: '0:10' },
  { note: 'C', octave: 5, duration: '𝅗𝅥', timestamp: '0:12' },
];

const NoteIcon = ({ index }: { index: number }) => {
  const icons = [Music, Music2, Music3, Music4];
  const Icon = icons[index % icons.length];
  return <Icon className="w-4 h-4" />;
};

const getQuadrantStrings = (slot: AlternatingSlotEntryPayload | null, fallbackPrimaryQuadrant: 'primary' | 'secondary') => {
  const strings = Array.isArray(slot?.strings)
    ? slot!.strings.filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
    : [];
  const primary = strings[0] ?? 0;
  const secondary = strings[1] ?? null;
  return {
    primary,
    secondary,
    showPrimaryDot: Boolean(slot?.left_hand_involved),
    showSecondaryDot: Boolean(slot?.left_hand_involved && secondary !== null),
    fallbackPrimaryQuadrant,
  };
};

const buildAlternatingPairs = (payload: AlternatingOnOffSlotsPayload | null | undefined): AlternatingPairCell[] => {
  const seq = Array.isArray(payload?.sequence) ? payload!.sequence : [];
  const normalized: AlternatingSlotCell[] = [];
  seq.forEach((row) => {
    if (!row || typeof row !== 'object') return;
    if ('on_beat' in row && row.on_beat && typeof row.on_beat === 'object') {
      normalized.push({ kind: 'on_beat', data: row.on_beat as AlternatingSlotEntryPayload });
      return;
    }
    if ('off_beat' in row && row.off_beat && typeof row.off_beat === 'object') {
      normalized.push({ kind: 'off_beat', data: row.off_beat as AlternatingSlotEntryPayload });
    }
  });

  if (normalized.length === 0) {
    return [];
  }

  const pairs: AlternatingPairCell[] = [];
  let current: AlternatingPairCell = { onBeat: null, offBeat: null };
  normalized.forEach((slot) => {
    if (slot.kind === 'on_beat') {
      if (current.onBeat !== null || current.offBeat !== null) {
        pairs.push(current);
        current = { onBeat: null, offBeat: null };
      }
      current.onBeat = slot.data;
      return;
    }
    if (current.offBeat !== null) {
      pairs.push(current);
      current = { onBeat: null, offBeat: null };
    }
    current.offBeat = slot.data;
  });
  if (current.onBeat !== null || current.offBeat !== null) {
    pairs.push(current);
  }
  return pairs;
};

const StringMark = ({ value, dotted }: { value: number | null; dotted: boolean }) => {
  if (value === null) return null;
  return (
    <span className="inline-flex items-center gap-1 font-heading text-base font-bold leading-none">
      <span>{value}</span>
      {dotted && value !== 0 ? (
        <span className="h-1.5 w-1.5 rounded-full bg-cyan-300 shadow-[0_0_8px_rgba(103,232,249,0.6)]" />
      ) : null}
    </span>
  );
};

const MusicalNotesDisplay = ({ isVisible, alternatingSlots }: MusicalNotesDisplayProps) => {
  const [visibleStrings, setVisibleStrings] = useState<number[]>([]);
  const [visibleNotes, setVisibleNotes] = useState<number[]>([]);
  const notePairs = buildAlternatingPairs(alternatingSlots);
  const hasAlternatingGrid = notePairs.length > 0;

  useEffect(() => {
    if (!isVisible) {
      setVisibleStrings([]);
      setVisibleNotes([]);
      return;
    }

    // Stagger string appearances
    mockStrings.forEach((_, i) => {
      setTimeout(() => {
        setVisibleStrings(prev => [...prev, i]);
      }, i * 150);
    });

    // Then stagger note appearances
    setTimeout(() => {
      mockNotes.forEach((_, i) => {
        setTimeout(() => {
          setVisibleNotes(prev => [...prev, i]);
        }, i * 100);
      });
    }, mockStrings.length * 150 + 300);
  }, [isVisible]);

  if (!isVisible) return null;

  return (
    <div className={`w-full mx-auto grid gap-6 animate-fade-in ${hasAlternatingGrid ? 'max-w-screen-2xl grid-cols-1' : 'max-w-6xl md:grid-cols-2'}`}>
      {!hasAlternatingGrid ? (
        /* Detected Strings */
        <div className="glass-strong rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-primary/20 flex items-center justify-center">
              <Music2 className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h3 className="font-heading text-lg font-bold">Detected Strings</h3>
              <p className="text-xs text-muted-foreground">Identified vibrating strings</p>
            </div>
          </div>

          <div className="space-y-3">
            {mockStrings.map((string, i) => (
              <div
                key={string.id}
                className={`glass p-3 rounded-xl transition-all duration-500 ${
                  visibleStrings.includes(i)
                    ? 'opacity-100 translate-x-0'
                    : 'opacity-0 -translate-x-4'
                }`}
                style={{ transitionDelay: `${i * 50}ms` }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div 
                      className="w-1 h-8 rounded-full"
                      style={{
                        background: 'var(--gradient-primary)',
                        boxShadow: 'var(--shadow-glow-sm)',
                      }}
                    />
                    <div>
                      <p className="text-sm font-medium">{string.name}</p>
                      <p className="text-xs text-muted-foreground">{string.frequency}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-16 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-1000 ease-out"
                        style={{
                          width: `${string.confidence}%`,
                          background: 'var(--gradient-primary)',
                          transitionDelay: `${i * 100 + 300}ms`,
                        }}
                      />
                    </div>
                    <span className="text-xs text-primary font-medium">{string.confidence}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {/* Generated Notes */}
      <div className="glass-strong rounded-2xl p-6">
        <div className="flex items-center gap-3 mb-6">
          <div 
            className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ 
              background: 'hsl(var(--gold) / 0.2)',
            }}
          >
            <Music className="w-5 h-5 text-gold" />
          </div>
          <div>
            <h3 className="font-heading text-lg font-bold">Generated Notes</h3>
            <p className="text-xs text-muted-foreground">Musical notation sequence</p>
          </div>
        </div>

        {notePairs.length > 0 ? (
          <>
            <div className="mb-3 flex items-center justify-between gap-3 text-xs text-muted-foreground">
              <span>Each card = one on/off beat pair</span>
              <span>Dot on number = left hand involved in that slot</span>
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-4 xl:grid-cols-6 2xl:grid-cols-8">
              {notePairs.map((pair, i) => {
                const on = getQuadrantStrings(pair.onBeat, 'primary');
                const off = getQuadrantStrings(pair.offBeat, 'primary');
                const onBeatIndex = typeof pair.onBeat?.beat_index === 'number' ? pair.onBeat.beat_index : null;
                const offBeatIndex = typeof pair.offBeat?.beat_index === 'number' ? pair.offBeat.beat_index : null;
                const label = onBeatIndex !== null || offBeatIndex !== null
                  ? `${onBeatIndex !== null ? onBeatIndex : '-'} / ${offBeatIndex !== null ? offBeatIndex : '-'}`
                  : `Pair ${i + 1}`;
                const isVisibleCard = visibleNotes.includes(i) || i >= mockNotes.length;
                return (
                  <div
                    key={`pair-${i}-${onBeatIndex ?? 'x'}-${offBeatIndex ?? 'y'}`}
                    className={`relative glass rounded-xl p-3 transition-all duration-500 ${
                      isVisibleCard ? 'opacity-100 scale-100' : 'opacity-0 scale-90'
                    }`}
                    style={{ transitionDelay: `${Math.min(i, 16) * 45}ms` }}
                  >
                    <div className="mb-2 flex items-center justify-between text-[11px] text-muted-foreground">
                      <span>Pair {i + 1}</span>
                      <span>{label}</span>
                    </div>

                    <div className="relative h-28 rounded-lg border border-border/30 bg-black/10">
                      <div className="absolute left-1/2 top-1 bottom-1 border-l border-dashed border-white/20" />
                      <div className="absolute top-1/2 left-1 right-1 border-t border-dashed border-white/20" />

                      <div className="absolute left-2 top-2 flex items-center">
                        <StringMark value={on.primary} dotted={on.showPrimaryDot} />
                      </div>
                      <div className="absolute left-2 bottom-2 flex items-center">
                        <StringMark value={on.secondary} dotted={on.showSecondaryDot} />
                      </div>
                      <div className="absolute right-2 top-2 flex items-center justify-end">
                        <StringMark value={off.secondary} dotted={Boolean(pair.offBeat?.left_hand_involved && off.secondary !== null)} />
                      </div>
                      <div className="absolute right-2 bottom-2 flex items-center justify-end">
                        <StringMark value={off.primary} dotted={Boolean(pair.offBeat?.left_hand_involved)} />
                      </div>

                      <div className="pointer-events-none absolute inset-x-0 top-1/2 -translate-y-1/2 text-center text-[10px] text-white/30">
                        on | off
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="mt-4 border-t border-border/30 pt-3 text-[11px] text-muted-foreground">
              TL=On1, BL=On2, BR=Off1, TR=Off2
            </div>
          </>
        ) : (
          <div className="rounded-xl border border-border/40 bg-background/20 p-4 text-sm text-muted-foreground">
            No alternating on/off slot data returned yet. Run a prediction to generate the note grid.
          </div>
        )}
      </div>
    </div>
  );
};

export default MusicalNotesDisplay;
