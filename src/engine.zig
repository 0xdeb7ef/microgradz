const std = @import("std");

const ExprType = enum {
    nop,
    unary,
    binary,
};

const UnaryType = enum {
    tanh,
    exp,
    relu,
};

const BinaryType = enum {
    add,
    sub,
    mul,
    pow,
};

pub fn Value(comptime T: type) type {
    if (@typeInfo(T) != .Float) {
        @compileError("Only floating-point types are supported.");
    }

    return struct {
        const Self = @This();
        const Expr = union(ExprType) {
            nop: void,
            unary: struct {
                op: UnaryType,
                args: [1]*Self,
            },
            binary: struct {
                op: BinaryType,
                args: [2]*Self,
            },
        };

        data: T,
        grad: T,
        expr: Expr,

        var arena: std.heap.ArenaAllocator = undefined;

        pub fn init(alloc: std.mem.Allocator) void {
            arena = std.heap.ArenaAllocator.init(alloc);
        }

        pub fn deinit() void {
            arena.deinit();
        }

        pub fn new(value: T) *Self {
            return create(value, Expr{ .nop = {} });
        }

        fn create(value: T, expr: Expr) *Self {
            const v = arena.allocator().create(Self) catch unreachable;

            v.* = Self{ .data = value, .grad = 0, .expr = expr };

            return v;
        }

        // Backprop //
        fn backprop_value(self: *Self) void {
            // TODO - Maybe switch to function pointers?
            switch (self.expr) {
                .nop => {},
                .unary => |u| {
                    switch (u.op) {
                        .tanh => tanh_back(self),
                        .exp => exp_back(self),
                        .relu => relu_back(self),
                    }
                },
                .binary => |b| {
                    switch (b.op) {
                        .add => add_back(self),
                        .sub => sub_back(self),
                        .mul => mul_back(self),
                        .pow => pow_back(self),
                    }
                },
            }
        }

        // Operations //
        fn unary(value: T, op: UnaryType, arg0: *Self) *Self {
            return create(value, Expr{
                .unary = .{
                    .op = op,
                    .args = [1]*Self{arg0},
                },
            });
        }

        fn binary(value: T, op: BinaryType, arg0: *Self, arg1: *Self) *Self {
            return create(value, Expr{
                .binary = .{
                    .op = op,
                    .args = [2]*Self{ arg0, arg1 },
                },
            });
        }

        pub fn add(self: *Self, other: anytype) *Self {
            switch (@TypeOf(other)) {
                *Self => {
                    return binary(self.data + other.data, .add, self, other);
                },
                else => {
                    const o = new(other);
                    return binary(self.data + o.data, .add, self, o);
                },
            }
        }
        fn add_back(self: *Self) void {
            self.expr.binary.args[0].grad += self.grad;
            self.expr.binary.args[1].grad += self.grad;
        }

        pub fn sub(self: *Self, other: anytype) *Self {
            switch (@TypeOf(other)) {
                *Self => {
                    return binary(self.data - other.data, .sub, self, other);
                },
                else => {
                    const o = new(other);
                    return binary(self.data - o.data, .sub, self, o);
                },
            }
        }
        fn sub_back(self: *Self) void {
            self.expr.binary.args[0].grad += self.grad;
            self.expr.binary.args[1].grad -= self.grad;
        }

        pub fn div(self: *Self, other: anytype) *Self {
            switch (@TypeOf(other)) {
                *Self => {
                    return self.mul(other.pow(-1));
                },
                else => {
                    return self.mul(new(other).pow(-1));
                },
            }
        }

        pub fn pow(self: *Self, other: anytype) *Self {
            switch (@TypeOf(other)) {
                *Self => {
                    switch (other.expr) {
                        .nop => {
                            return binary(std.math.pow(T, self.data, other.data), .pow, self, other);
                        },
                        else => {
                            @panic("Only Values with nop type are supported.");
                        },
                    }
                },
                else => |s| {
                    switch (@typeInfo(s)) {
                        .Int, .Float, .ComptimeInt, .ComptimeFloat => {
                            const o = new(other);
                            return binary(std.math.pow(T, self.data, o.data), .pow, self, o);
                        },
                        else => {
                            @compileError("Only integers and floating-point types are supported.");
                        },
                    }
                },
            }
        }
        fn pow_back(self: *Self) void {
            const selfData = self.expr.binary.args[0].data;
            const otherData = self.expr.binary.args[1].data;

            self.expr.binary.args[0].grad += otherData * std.math.pow(T, selfData, otherData - 1) * self.grad;
        }

        pub fn mul(self: *Self, other: anytype) *Self {
            switch (@TypeOf(other)) {
                *Self => {
                    return binary(self.data * other.data, .mul, self, other);
                },
                else => {
                    const o = new(other);
                    return binary(self.data * o.data, .mul, self, o);
                },
            }
        }
        fn mul_back(self: *Self) void {
            self.expr.binary.args[0].grad += self.grad * self.expr.binary.args[1].data;
            self.expr.binary.args[1].grad += self.grad * self.expr.binary.args[0].data;
        }

        pub fn tanh(self: *Self) *Self {
            return unary((@exp(self.data * 2) - 1) / (@exp(self.data * 2) + 1), .tanh, self);
        }
        fn tanh_back(self: *Self) void {
            self.expr.unary.args[0].grad += (1 - self.data * self.data) * self.grad;
        }

        pub fn exp(self: *Self) *Self {
            return unary(@exp(self.data), .exp, self);
        }
        fn exp_back(self: *Self) void {
            self.expr.unary.args[0].grad += @exp(self.expr.unary.args[0].data) * self.grad;
        }

        pub fn relu(self: *Self) *Self {
            return unary(@max(0, self.data), .relu, self);
        }
        fn relu_back(self: *Self) void {
            const grad = if (self.data > 0) self.grad else 0;
            self.expr.unary.args[0].grad += grad;
        }
    };
}

pub fn Backprop(comptime T: type) type {
    return struct {
        const Self = @This();

        topo: std.ArrayList(*T),
        visited: std.AutoHashMap(*const T, void),

        pub fn init(alloc: std.mem.Allocator) Self {
            return Self{
                .topo = std.ArrayList(*T).init(alloc),
                .visited = std.AutoHashMap(*const T, void).init(alloc),
            };
        }

        pub fn deinit(self: *Self) void {
            self.topo.deinit();
            self.visited.deinit();
        }

        // Backprop //
        pub fn backprop(self: *Self, root: *T) void {
            self.topo.clearRetainingCapacity();
            self.visited.clearRetainingCapacity();
            self.backwards_rec(root);

            root.grad = 1;
            var top = self.topo.items;
            for (self.topo.items, 0..) |_, idx| {
                top[top.len - idx - 1].backprop_value();
            }
        }

        fn backwards_rec(self: *Self, root: *T) void {
            const res = self.visited.getOrPut(root) catch unreachable;

            if (!res.found_existing) {
                switch (root.expr) {
                    .nop => {},
                    .unary => |u| {
                        backwards_rec(self, u.args[0]);
                    },
                    .binary => |b| {
                        backwards_rec(self, b.args[0]);
                        backwards_rec(self, b.args[1]);
                    },
                }

                self.topo.append(root) catch unreachable;
            }
        }
    };
}

// TODO - Add tests.
