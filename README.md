# Micrograd in Zig

Inspired by [Micrograd](https://github.com/karpathy/micrograd) and [Zigrograd](https://github.com/nurpax/zigrograd) (you will notice some similarities).

Its only purpose is to serve as a toy example to follow along with Andrej Karpathy's videos.

## Example

```zig
// import Value and Backprop
const ValueType = @import("engine.zig").Value;
const BackpropType = @import("engine.zig").Backprop;

// ...

// Create the Value pool with a specific float type
const Value = ValueType(f32);
Value.init(alloc); // init with an allocator, arena is used internally
defer Value.deinit();

// Create a reusable Backprop instance that operates on Value
var Backprop = BackpropType(Value).init(alloc); // init with an allocator
defer Backprop.deinit();

var a = Value.new(-4);
var b = Value.new(2);
var c = a.add(b);
var d = a.mul(b).add(b.pow(3));
// c += c + 1
c = c.add(c.add(1));
// c += 1 + c + (-a)
c = c.add(c.add(1).sub(a));
// d += d * 2 + (b + a).relu()
d = d.add(d.mul(2).add(b.add(a).relu()));
// d += 3 * d + (b - a).relu()
d = d.add(d.mul(3).add(b.sub(a).relu()));

var e = c.sub(d);
var f = e.pow(2);
var g = f.div(2);
// g += 10.0 / f
g = g.add(Value.new(10).div(f));

print("g.data: {d:.4}\n", .{g.data}); // 24.7041

// run backprop on `g`
Backprop.backprop(g);

print("a.grad: {d:.4}\n", .{a.grad}); // 138.8338
print("b.grad: {d:.4}\n", .{b.grad}); // 645.5773
```

## Training

The code in `main.zig` is a sample toy example that can be run with:

```console
zig build run
```

## Running tests

Currently there are no tests implemented, I plan on adding them.

`zig build test` should do the trick, regardless.
