import React, { useCallback, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { GraphNode, GraphEdge } from "../api/api-client";

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

// Colors per node type
const NODE_COLOR: Record<string, string> = {
  session: "#aa3bff",   // primary purple
  chunk: "#71717a",     // zinc-500
  entity: "#10b981",    // emerald-500
};

const NODE_RADIUS: Record<string, number> = {
  session: 14,
  chunk: 10,
  entity: 7,
};

const LINK_COLOR: Record<string, string> = {
  CONTAINS: "#a1a1aa",   // zinc-400
  MENTIONS: "#6ee7b7",   // emerald-300
  ALIAS_OF: "#c4b5fd",   // violet-300
};

export function GraphView({ nodes, edges }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  // react-force-graph-2d calls these "links" and mutates them in place
  // so we must create fresh plain objects every render
  const graphData = React.useMemo(
    () => ({
      nodes: nodes.map((n) => ({ ...n })),
      links: edges.map((e) => ({ ...e })),
    }),
    [nodes, edges]
  );

  const nodeCanvasObject = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const r = NODE_RADIUS[node.node_type] ?? 8;
      const label: string = node.label ?? node.id;
      const fontSize = Math.max(10 / globalScale, 4);

      // Circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
      ctx.fillStyle = NODE_COLOR[node.node_type] ?? "#71717a";
      ctx.fill();

      // Border ring
      ctx.strokeStyle = "rgba(255,255,255,0.3)";
      ctx.lineWidth = 1.5 / globalScale;
      ctx.stroke();

      // Label below
      ctx.font = `${fontSize}px Inter, system-ui, sans-serif`;
      ctx.fillStyle = "rgba(244,244,245,0.9)";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";

      // Wrap long labels
      const maxChars = 18;
      const display = label.length > maxChars ? label.slice(0, maxChars) + "…" : label;
      ctx.fillText(display, node.x, node.y + r + 2 / globalScale);
    },
    []
  );

  const linkColor = useCallback(
    (link: any) => LINK_COLOR[link.relation] ?? "#a1a1aa",
    []
  );

  const linkLabel = useCallback((link: any) => link.relation, []);

  if (nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-zinc-500 text-sm">
        No graph data available.
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="w-full rounded-xl overflow-hidden border border-zinc-800 bg-zinc-950"
      style={{ height: 340 }}
    >
      <ForceGraph2D
        graphData={graphData}
        width={containerRef.current?.clientWidth ?? 700}
        height={340}
        backgroundColor="#09090b"
        nodeCanvasObject={nodeCanvasObject}
        nodeCanvasObjectMode={() => "replace"}
        linkColor={linkColor}
        linkLabel={linkLabel}
        linkWidth={1.2}
        linkDirectionalArrowLength={5}
        linkDirectionalArrowRelPos={1}
        cooldownTicks={120}
        nodeRelSize={6}
      />
      {/* Legend */}
      <div className="flex items-center gap-4 px-4 py-2 border-t border-zinc-800 text-xs text-zinc-400">
        {Object.entries(NODE_COLOR).map(([type, color]) => (
          <span key={type} className="flex items-center gap-1.5">
            <span
              className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0"
              style={{ background: color }}
            />
            {type}
          </span>
        ))}
      </div>
    </div>
  );
}
