from qrisp import QuantumVariable, x, control, control
from qrisp.jasp import q_while_loop, q_cond
import jax.numpy as jnp


def tree_switch_tree_jasp(operand, case, case_fun):
    n = case.size

    B_ID = 1
    D_ID = 2
    U_ID = 3
    L_ID = 4

    def bounce(d: int, anc, ca, oper):
        with control(anc[d-1]):
            x(anc[d])
        with control(anc[d]):
            x(anc[d+1])
        with control(anc[d-1]):
            with control(ca[n-1-d]):
                x(anc[d+1])

    def down(d: int, anc, ca, oper):
        with control(anc[d]):
            x(ca[n-1-d])
            with control(ca[n-1-d]):
                x(anc[d+1])
            x(ca[n-1-d])

    def up(d: int, anc, ca, oper):
        with control(anc[d]):
            with control(ca[n-1-d]):
                x(anc[d+1])

    def leaf(d: int, A, B, anc, ca, oper):
        with control(anc[d+1]):
            A(oper)
        with control(anc[d]):
            x(anc[d+1])
        with control(anc[d+1]):
            B(oper)

    def f1(stack, depth):
        stack = jnp.roll(stack, shift=1, axis=0)
        stack = stack.at[0].set((L_ID, depth))
        return stack

    def f2(stack, depth):
        stack = jnp.roll(stack, shift=3, axis=0)
        stack = stack.at[0].set((D_ID, depth+1))
        stack = stack.at[1].set((B_ID, depth+1))
        stack = stack.at[2].set((U_ID, depth+1))
        return stack

    def cond_fun(val):
        stack, cfl_pointer, anc, ca, oper = val
        return jnp.sum(stack[:, 0] != 0) > 0

    def body_fun(val):
        stack, cfl_pointer, anc, ca, oper = val
        op, depth = stack[0]
        stack = stack.at[0].set((0, 0))
        stack = jnp.roll(stack, shift=-1, axis=0)

        q_cond(op == B_ID, bounce, lambda a, b, c, d: None, depth, anc, ca, oper)
        q_cond(op == D_ID, down, lambda a, b, c, d: None, depth, anc, ca, oper)
        q_cond(op == U_ID, up, lambda a, b, c, d: None, depth, anc, ca, oper)

        q_cond(op == L_ID, leaf, lambda a, b, c, d, e, f: None, depth,
               lambda args: case_fun(cfl_pointer, args),
               lambda args: case_fun(cfl_pointer + 1, args), anc, ca, oper)

        cfl_pointer = q_cond(op == L_ID, lambda: cfl_pointer + 2, lambda: cfl_pointer)

        q = jnp.logical_or(op == D_ID, op == B_ID)
        stack = q_cond(jnp.logical_and(q, depth + 1 >= n),
                       f1, (lambda a, b: a), stack, depth)
        stack = q_cond(jnp.logical_and(q, depth + 1 < n),
                       f2, (lambda a, b: a), stack, depth)

        return stack, cfl_pointer, anc, ca, oper

    stack_s = jnp.zeros((200, 2), dtype=jnp.int64)  # Just set stack large enough

    stack_s = stack_s.at[0].set((D_ID, 1))
    stack_s = stack_s.at[1].set((B_ID, 1))
    stack_s = stack_s.at[2].set((U_ID, 1))

    anc = QuantumVariable(n+1)
    x(anc[0])

    down(0, anc, case, operand)
    stack, cfl_pointer, anc, case, operand = q_while_loop(
        cond_fun, body_fun, (stack_s, 0, anc, case, operand))
    up(0, anc, case, operand)
    return case, operand
