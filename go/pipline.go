package main

import (
	"context"
	"log"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()
	if err := run(ctx); err != nil {
		log.Fatalf("❌ Dagger pipeline failed: %v", err)
	}
	log.Println("✅ Dagger pipeline completed successfully!")
}

func run(ctx context.Context) error {
	// 1️⃣ Connect to Dagger
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(log.Writer()))
	if err != nil {
		return err
	}
	defer client.Close()

	// 2️⃣ Python container running main.py from repo root
	python := client.Container().
		From("python:3.10").
		WithMountedDirectory("/src", client.Host().Directory(".")).
		WithWorkdir("/src").
		WithExec([]string{"python", "-m", "pip", "install", "--upgrade", "pip"}).
		WithExec([]string{"pip", "install", "-r", "notebooks/requirements.txt"}).
		WithExec([]string{"python", "main.py"})

	// 3️⃣ Sync outputs (artifacts, deployment, model file) back to host
	_, err = python.Sync(ctx)
	return err
}
