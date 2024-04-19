const std = @import("std");

pub fn Neuron(comptime ValueType: type, comptime R: type) type {
    return struct {
        const Self = @This();

        weights: []*ValueType,
        bias: *ValueType,

        var arena: std.heap.ArenaAllocator = undefined;
        const random = Random(R);

        pub fn init(alloc: std.mem.Allocator) void {
            arena = std.heap.ArenaAllocator.init(alloc);
            random.set_seed(@as(u64, @bitCast(std.time.milliTimestamp())));
        }

        pub fn deinit() void {
            arena.deinit();
        }

        pub fn new(inputs: usize) *Self {
            const norm = 1 / @sqrt(@as(f32, @floatFromInt(inputs)));
            const n = arena.allocator().create(Self) catch unreachable;
            const w = arena.allocator().alloc(*ValueType, inputs) catch unreachable;

            var j: usize = 0;
            while (j < inputs) : (j += 1) {
                w[j] = ValueType.new((random.uniform() * 2 - 1) * norm);
            }

            n.* = Self{
                .weights = w[0..],
                .bias = ValueType.new((random.uniform() * 2 - 1) * norm),
            };

            return n;
        }

        pub fn forward(self: *Self, inputs: []*ValueType) *ValueType {
            if (inputs.len != self.weights.len) {
                @panic("Input length is invalid.");
            }

            var sum = self.bias;
            for (self.weights, inputs) |w, x| {
                sum = sum.add(w.mul(x));
            }

            return sum;
            // FIXME - adding tanh completely breaks it.
            // No idea why.
            // return sum.tanh();
        }

        pub fn parameters(self: *Self) []*ValueType {
            var list = std.ArrayList(*ValueType).init(arena.allocator());
            defer list.deinit();

            for (self.weights) |w| {
                list.append(w) catch unreachable;
            }
            list.append(self.bias) catch unreachable;

            return list.toOwnedSlice() catch unreachable;
        }
    };
}

pub fn Layer(comptime NeuronType: type, comptime ValueType: type) type {
    return struct {
        const Self = @This();

        neurons: []*NeuronType,

        var arena: std.heap.ArenaAllocator = undefined;

        pub fn init(alloc: std.mem.Allocator) void {
            arena = std.heap.ArenaAllocator.init(alloc);
        }

        pub fn deinit() void {
            arena.deinit();
        }

        pub fn new(inputs: usize, outputs: usize) *Self {
            const l = arena.allocator().create(Self) catch unreachable;
            const n = arena.allocator().alloc(*NeuronType, outputs) catch unreachable;

            var i: usize = 0;
            while (i < outputs) : (i += 1) {
                n[i] = NeuronType.new(inputs);
            }

            l.* = Self{
                .neurons = n[0..],
            };

            return l;
        }

        pub fn forward(self: *Self, inputs: []*ValueType) []*ValueType {
            var list = std.ArrayList(*ValueType).init(arena.allocator());
            defer list.deinit();

            for (self.neurons) |n| {
                list.append(n.forward(inputs)) catch unreachable;
            }

            return list.toOwnedSlice() catch unreachable;
        }

        pub fn parameters(self: *Self) []*ValueType {
            var list = std.ArrayList(*ValueType).init(arena.allocator());
            defer list.deinit();

            for (self.neurons) |n| {
                list.appendSlice(n.parameters()) catch unreachable;
            }

            return list.toOwnedSlice() catch unreachable;
        }
    };
}

pub fn MLP(comptime LayerType: type, comptime ValueType: type) type {
    return struct {
        const Self = @This();

        layers: []*LayerType,

        var arena: std.heap.ArenaAllocator = undefined;

        pub fn init(alloc: std.mem.Allocator) void {
            arena = std.heap.ArenaAllocator.init(alloc);
        }

        pub fn deinit() void {
            arena.deinit();
        }

        pub fn new(inputs: usize, layers: []usize) *Self {
            const m = arena.allocator().create(Self) catch unreachable;
            const l = arena.allocator().alloc(*LayerType, layers.len) catch unreachable;

            l[0] = LayerType.new(inputs, layers[0]);
            var i: usize = 0;
            while (i < layers.len - 1) : (i += 1) {
                l[i + 1] = LayerType.new(layers[i], layers[i + 1]);
            }

            m.* = Self{
                .layers = l[0..],
            };

            return m;
        }

        pub fn forward(self: *Self, inputs: []*ValueType) []*ValueType {
            var x = inputs;
            for (self.layers) |l| {
                x = l.forward(x);
            }

            return x;
        }

        pub fn parameters(self: *Self) []*ValueType {
            var list = std.ArrayList(*ValueType).init(arena.allocator());
            defer list.deinit();

            for (self.layers) |l| {
                list.appendSlice(l.parameters()) catch unreachable;
            }

            return list.toOwnedSlice() catch unreachable;
        }
    };
}

pub fn Random(comptime T: type) type {
    return struct {
        const Self = @This();

        var random = std.Random.DefaultPrng.init(0x0);

        pub fn set_seed(seed: u64) void {
            random = std.Random.DefaultPrng.init(seed);
        }

        pub fn uniform() T {
            return switch (@typeInfo(T)) {
                .Float => random.random().float(T),
                .Int => random.random().int(T),
                else => {
                    @compileError("Only integer and floating-point types are supported.");
                },
            };
        }
    };
}
