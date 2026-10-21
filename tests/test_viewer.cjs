// Numerical viewer contracts; no browser or measurement data required.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const context = vm.createContext({
  document: {getElementById: () => ({addEventListener() {}})},
  window: {location: {search: ''}, addEventListener() {}},
  URLSearchParams,
});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../viewer/app.js'), 'utf8'), context);
const plain = value => JSON.parse(JSON.stringify(value));

test('extinction colors are independent of displayed sample count', () => {
  const group = {out1: [0, 1, .9], out2: [0, 1, .3], out3: [1, 0, .2]};
  const full = plain(context.extinction(group, [0, 1, 2]));
  const subset = plain(context.extinction(group, [2]));
  assert.deepEqual(subset[0], full[2]);
});

test('constant channels produce finite neutral power and zero extinction', () => {
  const group = {out1: [1, 1], out2: [2, 2], out3: [3, 3]};
  assert.deepEqual(plain(context.extinction(group, [0, 1])), [[0, 0, 0], [0, 0, 0]]);
  assert.deepEqual(plain(context.normalizedOutputs(group, [0], [[1, 1], [2, 2], [3, 3]])), [[.5, .5, .5]]);
});
