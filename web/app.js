import init, { WasmProgram } from './pkg/axm_rs.js';

const log = (msg) => {
  document.getElementById('log').textContent = msg;
};

async function loadFile(file) {
  const buffer = await file.arrayBuffer();
  await init();
  return new WasmProgram(new Uint8Array(buffer));
}

document.getElementById('runQuery').addEventListener('click', async () => {
  const input = document.getElementById('fileInput');
  if (!input.files.length) {
    log('Please choose a .axm zip first.');
    return;
  }
  try {
    const program = await loadFile(input.files[0]);
    const manifest = program.manifest();
    const nodes = program.query(7, undefined, undefined, undefined, undefined, undefined);
    const neighborSummary = nodes.length
      ? program.neighbors(nodes[0].id, 1.0).length
      : 0;
    log(`Loaded program with ${manifest.counts.nodes} nodes\nFound ${nodes.length} quantity nodes; first node has ${neighborSummary} neighbors within radius 1.0.`);
  } catch (err) {
    console.error(err);
    log(`Failed to load program: ${err}`);
  }
});
