
from jax import core, make_jaxpr

enter_inv_env_p = core.Primitive("enter_inv")
exit_inv_env_p = core.Primitive("exit_inv")

class DummyValue(core.AbstractValue):
    pass

core.raise_to_shaped_mappings[DummyValue] = lambda aval, _: aval

dummy_fun = lambda : DummyValue()

enter_inv_env_p.def_abstract_eval(dummy_fun)
exit_inv_env_p.def_abstract_eval(dummy_fun)

#%%

def get_adjoint(eqn_list):
    # Put more general implementation here
    return eqn_list[::-1]


mlir_implementation_available = {"inv" : False}

def compile_inv_environments(closed_jaxpr):
    
    jaxpr = closed_jaxpr.jaxpr
    
    environment_stack = [[]]
    # Loop through equations and compile inversion environments accordingly
    for eqn in jaxpr.eqns:
        
        op_name = eqn.primitive.name
        
        if op_name == "enter_inv" and not mlir_implementation_available[op_name.split("_")[1]]:
            environment_stack.append([])
        elif op_name == "exit_inv" and not mlir_implementation_available[op_name.split("_")[1]]:
            content = environment_stack.pop(-1)
            inv_content = get_adjoint(content)
            
            environment_stack[-1].extend(inv_content)
        else:
            environment_stack[-1].append(eqn)
    
    return core.Jaxpr(closed_jaxpr.consts, jaxpr.invars, jaxpr.outvars, environment_stack[0])


class InversionEnvironment:
    
    def __enter__(self):
        enter_inv_env_p.bind()    
        
    def __exit__(self, exception_type, exception_value, traceback):
        exit_inv_env_p.bind()


#%%        

def apply_op(name, qb):
    return qinst_p.bind(qb, op = name, qubits_len = 1)[0]


from catalyst.jax_primitives import qalloc_p, qextract_p, qinst_p
def test_f():
    
    reg = qalloc_p.bind(1)
    qb = qextract_p.bind(reg, 0)
    qb = apply_op("Hadamard", qb)
    
    with InversionEnvironment():
        qb = apply_op("PauliX", qb)
        qb = apply_op("PauliY", qb)
    
    qb = apply_op("PauliZ", qb)
    
    
test_jaxpr = make_jaxpr(test_f)()

print("Original jaxpr:")
print(test_jaxpr)

print(60*"=")
print("Compiled jaxpr:")
inv_test_jaxpr = compile_inv_environments(test_jaxpr)

print(inv_test_jaxpr)

    
    
    
