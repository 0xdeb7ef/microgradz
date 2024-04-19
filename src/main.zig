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

    // init the Value pool with f64 type
    const Value = ValueType(f64);
    Value.init(alloc);
    defer Value.deinit();

    // init the Backprop pool with the Value type
    var Backprop = BackpropType(Value).init(alloc);
    defer Backprop.deinit();

    // init the Neuron pool with the Value type and f64 random numbers
    const Neuron = NeuronType(Value, f64);
    Neuron.init(alloc);
    defer Neuron.deinit();

    // init the Layer pool with the Neuron and Value types
    const Layer = LayerType(Neuron, Value);
    Layer.init(alloc);
    defer Layer.deinit();

    // finally, init the MLP pool with the Layer and Value types
    const MLP = MLPType(Layer, Value);
    MLP.init(alloc);
    defer MLP.deinit();

    // pass the sizes as a slice
    // NOTE - might be better to refactor
    // the nn.zig code and use anytype instead?
    var sizes = [_]usize{ 4, 4, 1 };
    const M = MLP.new(3, &sizes);

    // inputs to the neural net
    const inputs = [4][3]*Value{
        [_]*Value{ Value.new(2), Value.new(3), Value.new(-1) },
        [_]*Value{ Value.new(3), Value.new(-1), Value.new(0.5) },
        [_]*Value{ Value.new(0.5), Value.new(1), Value.new(1) },
        [_]*Value{ Value.new(1), Value.new(1), Value.new(-1) },
    };

    // the desired output
    const outputs = [_]*Value{ Value.new(1), Value.new(-1), Value.new(-1), Value.new(1) };

    var loop: usize = 1;
    while (loop <= 20) : (loop += 1) {
        print("[{d:2}] ", .{loop});
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
}
