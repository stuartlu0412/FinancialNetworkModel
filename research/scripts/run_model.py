from src.model.erdos_renyi import ErdosRenyiModel

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Erdős-Rényi Market Model Simulation")
    print("="*70)
    
    # 1) Construct model with desired parameters
    print("\n1. Constructing model...")
    model = ErdosRenyiModel(
        alpha=20,
        beta=2,
        N=50000,
        k=4,
        seed=123
    )
    print("   Model constructed successfully!")

    # 2) Run simulation (do_plot=False for large networks)
    print("\n2. Running simulation...")
    df = model.simulate(
        frames=3000,
        output_dir="results/erdos_renyi",  # will be created if needed
        do_plot=False
    )

    # 3) df is a pandas.DataFrame of time series (M_t, F_t)
    print("\n3. Results:")
    print(df.head())
    print(f"\nFinal magnetization: {model.M_t}")
    print(f"Final fundamentalists: {model.F_t}")
    print(f"\nDataFrame shape: {df.shape}")
    print("\n" + "="*70)
    print("All done!")
    print("="*70 + "\n")