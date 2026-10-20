/**
 * One pass, drawn: the cortex (on disk), the membrane (a fixed budget), the value lock (at the edge of
 * your machine) and the model. Every string is from the README. A 10 s CSS loop (.dgm-* in globals.css);
 * with reduced motion, or before CSS animations run, the static drawing shows every beat at once.
 */
export function PassDiagram() {
  const ink = "#0b0b0f";
  const quiet = "#5d5d6a";
  const hair = "#d9d9e0";
  const pink = "#ec4a7f";
  const chip = (x: number, y: number, w: number, text: string, fill = "#fff", stroke = hair, color = ink) => (
    <g>
      <rect x={x} y={y} width={w} height={30} rx={15} fill={fill} stroke={stroke} />
      <text x={x + 14} y={y + 20} fontSize={13} fill={color}>{text}</text>
    </g>
  );
  const lock = (cy: number, cls: string, r: number) => {
    const k = r / 22;
    return (
      <g className={`dgm-lock ${cls}`}>
        <circle cx={700} cy={cy} r={r} fill="#fff" stroke={ink} strokeWidth={1.5} />
        <path d={`M${700 - 8 * k} ${cy - 1 * k}v${-5 * k}a${8 * k} ${8 * k} 0 0 1 ${16 * k} 0v${5 * k}`} fill="none" stroke={ink} strokeWidth={1.8 * k} />
        <rect x={700 - 11 * k} y={cy - 2 * k} width={22 * k} height={15 * k} rx={3 * k} fill={ink} />
      </g>
    );
  };
  return (
    <figure className="dgm-wrap" data-reveal="scale">
      <div className="dgm-scroll" tabIndex={0} role="group" aria-label="Diagram: one pass through Angular AI (scrolls sideways on small screens)">
        <svg viewBox="0 0 960 456" className="dgm" role="img" aria-labelledby="dgm-title dgm-desc">
          <title id="dgm-title">One pass through Angular AI</title>
          <desc id="dgm-desc">
            The question &ldquo;What color is the notebook?&rdquo; is ranked against the cortex on disk. &ldquo;The notebook is blue.&rdquo; shares its words and ranks 1; &ldquo;Catering is at noon.&rdquo; ranks 0 and stays unsent. Only what fits the fixed budget crosses the membrane into the workspace. The workspace goes to the model you already call. The reply comes back through the value lock and the pass is written to the file. A secret in the input is refused by the value lock before the call, and nothing is written.
          </desc>

          {/* zones */}
          <text x={24} y={28} fontSize={11} letterSpacing="1.6" fill={quiet}>YOUR MACHINE</text>
          <text x={936} y={28} fontSize={11} letterSpacing="1.6" fill={quiet} textAnchor="end">PROVIDER API</text>
          <line x1={700} y1={44} x2={700} y2={446} stroke={hair} strokeDasharray="4 6" />

          {/* cortex */}
          <rect x={24} y={96} width={300} height={226} rx={14} fill="#f6f6f8" stroke={hair} />
          <text x={44} y={124} fontSize={11} letterSpacing="1.6" fill={quiet}>CORTEX · ON DISK</text>
          <text x={304} y={124} fontSize={12} fill={quiet} textAnchor="end">passes.jsonl</text>
          <g className="dgm-l1">
            <rect x={40} y={140} width={268} height={34} rx={8} fill="#fff" stroke={hair} />
            <text x={54} y={162} fontSize={13.5} fill={ink}>The notebook is blue.</text>
          </g>
          <g>
            <rect x={40} y={182} width={268} height={34} rx={8} fill="#fff" stroke={hair} />
            <text x={54} y={204} fontSize={13.5} fill={ink}>Catering is at noon.</text>
          </g>
          <text className="dgm-rank" x={300} y={162} fontSize={12} fill={pink} textAnchor="end">rank 1</text>
          <text className="dgm-rank" x={300} y={204} fontSize={12} fill={quiet} textAnchor="end">rank 0</text>
          <text className="dgm-unsent" x={54} y={236} fontSize={12} fill={quiet}>unsent: the rest stays on disk</text>
          <g className="dgm-new">
            <rect x={40} y={252} width={268} height={34} rx={8} fill="#fff" stroke={pink} />
            <text x={54} y={274} fontSize={13} fill={ink}>+ one line: this pass</text>
          </g>
          <text x={44} y={308} fontSize={11.5} fill={quiet}>superseded lines stay in the file</text>

          {/* question */}
          <g className="dgm-q">{chip(24, 50, 236, "What color is the notebook?")}</g>

          {/* membrane */}
          <line x1={372} y1={84} x2={372} y2={334} stroke={pink} strokeWidth={2} strokeDasharray="2 7" strokeLinecap="round" className="dgm-membrane" />
          <text x={372} y={78} fontSize={11} letterSpacing="1.6" fill={ink} textAnchor="middle">MEMBRANE</text>

          {/* workspace */}
          <rect x={404} y={96} width={252} height={144} rx={14} fill="#fff" stroke={ink} />
          <text x={424} y={124} fontSize={11} letterSpacing="1.6" fill={quiet}>WORKSPACE · SENT</text>
          <rect x={424} y={206} width={212} height={10} rx={5} fill="#eeeef2" />
          <rect x={424} y={206} width={212} height={10} rx={5} fill={pink} className="dgm-meter" />
          <text x={424} y={232} fontSize={11.5} fill={quiet}>every call stays inside the budget</text>

          {/* packet: the ranked trace, copied into the workspace and on to the model */}
          <g className="dgm-pkt">
            <rect x={424} y={140} width={212} height={34} rx={8} fill="#fff" stroke={pink} />
            <text x={438} y={162} fontSize={13.5} fill={ink}>The notebook is blue.</text>
          </g>

          {/* value lock on the edge of the machine */}
          {lock(168, "dgm-lock-a", 22)}
          {lock(340, "dgm-lock-b", 15)}
          {lock(407, "dgm-lock-c", 15)}
          <text x={700} y={212} fontSize={11} letterSpacing="1.6" fill={ink} textAnchor="middle">VALUE LOCK</text>

          {/* model */}
          <rect x={744} y={96} width={192} height={144} rx={14} fill={ink} />
          <text x={764} y={124} fontSize={11} letterSpacing="1.6" fill="#b9b9c6">MODEL</text>
          <text x={764} y={162} fontSize={13.5} fill="#fff">The model you</text>
          <text x={764} y={182} fontSize={13.5} fill="#fff">already call.</text>
          <text x={764} y={218} fontSize={11.5} fill="#b9b9c6">weights unchanged</text>

          {/* reply lane */}
          <path d="M840 240 V 340 H 120" fill="none" stroke={hair} strokeDasharray="3 6" />
          <g className="dgm-reply">{chip(462, 325, 214, "The notebook is blue.", "#fff", ink)}</g>
          <text className="dgm-judged" x={722} y={370} fontSize={11.5} fill={quiet}>reply judged by the lock</text>

          {/* secret lane */}
          <g className="dgm-secret">{chip(462, 392, 210, "OPENAI_API_KEY=sk-••••", "#fff2f6", pink, "#c42a60")}</g>
          <text className="dgm-refused" x={722} y={404} fontSize={12} fill="#c42a60">a secret in the input: refused</text>
          <text className="dgm-refused" x={722} y={422} fontSize={12} fill="#c42a60">before the call; nothing written</text>
        </svg>
      </div>
      <figcaption className="mt-3 text-[13px] text-quiet">One pass. The words and numbers are the README&rsquo;s example. The loop repeats every ten seconds.</figcaption>
    </figure>
  );
}
