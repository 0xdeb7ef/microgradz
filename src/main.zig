const std = @import("std");
const print = std.debug.print;

const ValueType = @import("engine.zig").Value;
const BackpropType = @import("engine.zig").Backprop;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();
    defer _ = gpa.deinit();

    const Value = ValueType(f64);
    Value.init(alloc);
    defer Value.deinit();

    var Backprop = BackpropType(Value).init(alloc);
    defer Backprop.deinit();

    // inputs x1,x2
    const x1 = Value.new(2);
    const x2 = Value.new(0);

    // weights w1, w2
    const w1 = Value.new(-3);
    const w2 = Value.new(1);

    // bias b
    const b = Value.new(6.8813735870195432);

    const x1w1 = x1.mul(w1);
    const x2w2 = x2.mul(w2);
    const x1w1x2w2 = x1w1.add(x2w2);

    const n = x1w1x2w2.add(b);
    // const o = n.tanh();
    const e = n.mul(2).exp();
    const o = e.sub(1).div(e.add(1));

    Backprop.backprop(o);

    try std.json.stringify(o, .{}, std.io.getStdOut().writer());
    print("\n", .{});
}
