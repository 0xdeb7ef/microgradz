const std = @import("std");
const print = std.debug.print;

const ValueType = @import("engine.zig").Value;
const BackpropType = @import("engine.zig").Backprop;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();
    defer _ = gpa.deinit();

    const Value = ValueType(f32);
    Value.init(alloc);
    defer Value.deinit();

    var Backprop = BackpropType(Value).init(alloc);
    defer Backprop.deinit();

    // Example from Micrograd
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

    Backprop.backprop(g);

    print("a.grad: {d:.4}\n", .{a.grad}); // 138.8338
    print("b.grad: {d:.4}\n", .{b.grad}); // 645.5773

    // JSON output
    // try std.json.stringify(g, .{}, std.io.getStdOut().writer());
    // print("\n", .{});
}
