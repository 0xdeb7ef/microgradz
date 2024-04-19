const std = @import("std");
const print = std.debug.print;

const ValueType = @import("engine.zig").Value;
const BackpropType = @import("engine.zig").Backprop;
const NeuronType = @import("nn.zig").Neuron;
const LayerType = @import("nn.zig").Layer;
const MLPType = @import("nn.zig").MLP;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();
    defer _ = gpa.deinit();

    const Value = ValueType(f64);
    Value.init(alloc);
    defer Value.deinit();

    var Backprop = BackpropType(Value).init(alloc);
    defer Backprop.deinit();

    const Neuron = NeuronType(Value, f64);
    Neuron.init(alloc);
    defer Neuron.deinit();

    const Layer = LayerType(Neuron, Value);
    Layer.init(alloc);
    defer Layer.deinit();

    const MLP = MLPType(Layer, Value);
    MLP.init(alloc);
    defer MLP.deinit();

    var sizes = [_]usize{ 4, 4, 1 };
    const M = MLP.new(3, &sizes);

    const inputs = [4][3]*Value{
        [_]*Value{ Value.new(2), Value.new(3), Value.new(-1) },
        [_]*Value{ Value.new(3), Value.new(-1), Value.new(0.5) },
        [_]*Value{ Value.new(0.5), Value.new(1), Value.new(1) },
        [_]*Value{ Value.new(1), Value.new(1), Value.new(-1) },
    };

    const outputs = [_]*Value{ Value.new(1), Value.new(-1), Value.new(-1), Value.new(1) };

    var loop: usize = 1;
    while (loop <= 20) : (loop += 1) {
        print("[{d:3}] ", .{loop});
        var loss = Value.new(0);

        // Forward Pass
        for (inputs, 0..) |in, i| {
            const out = M.forward(@constCast(&in))[0];
            loss = loss.add(out.sub(outputs[i]).pow(2));
        }

        // Backprop
        const P = M.parameters();
        for (P) |p| {
            p.grad = 0;
        }
        Backprop.backprop(loss);

        // Update
        for (P) |p| {
            p.data -= 0.05 * p.grad;
        }

        print(" Loss: {d:.4}\n", .{loss.data});
    }

    print("\n", .{});
    print("Got     : ", .{});
    for (inputs) |in| {
        const out = M.forward(@constCast(&in));
        print("{d:7.4} ", .{out[0].data});
    }
    print("\n", .{});
    print("Expected: {d:7} {d:7} {d:7} {d:7}\n", .{
        outputs[0].data,
        outputs[1].data,
        outputs[2].data,
        outputs[3].data,
    });

    // print("{any}\n", .{P});

    // JSON output
    // try std.json.stringify(x, .{}, std.io.getStdOut().writer());
    // print("\n", .{});

    // Example from Micrograd
    // var a = Value.new(-4);
    // var b = Value.new(2);
    // var c = a.add(b);
    // var d = a.mul(b).add(b.pow(3));
    // // c += c + 1
    // c = c.add(c.add(1));
    // // c += 1 + c + (-a)
    // c = c.add(c.add(1).sub(a));
    // // d += d * 2 + (b + a).relu()
    // d = d.add(d.mul(2).add(b.add(a).relu()));
    // // d += 3 * d + (b - a).relu()
    // d = d.add(d.mul(3).add(b.sub(a).relu()));

    // var e = c.sub(d);
    // var f = e.pow(2);
    // var g = f.div(2);
    // // g += 10.0 / f
    // g = g.add(Value.new(10).div(f));

    // print("g.data: {d:.4}\n", .{g.data}); // 24.7041

    // Backprop.backprop(g);

    // print("a.grad: {d:.4}\n", .{a.grad}); // 138.8338
    // print("b.grad: {d:.4}\n", .{b.grad}); // 645.5773

    // JSON output
    // try std.json.stringify(g, .{}, std.io.getStdOut().writer());
    // print("\n", .{});
}
