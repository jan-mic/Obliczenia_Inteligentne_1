import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic

from captum.attr import Saliency, IntegratedGradients, FeatureAblation, Lime
from captum.attr import visualization as viz
from captum.robust import MinParamPerturbation
from captum._utils.models.linear_model import SkLearnLinearRegression


# ==============================================================================
# 1. FUNKCJE POMOCNICZE (DENORMALIZACJA)
# ==============================================================================
def unnormalize_imagenette(tensor):
    """Odwraca normalizację dla 3-kanałowego obrazu Imagenette i ustawia format (H, W, C)."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    img = tensor * std + mean
    img = img.clamp(0, 1).squeeze(0)
    img = img.permute(1, 2, 0)
    return img.detach().cpu().numpy()


def unnormalize_mnist(tensor):
    """Odwraca normalizację dla 1-kanałowego obrazu MNIST i ustawia format (H, W, 1)."""
    mean = torch.tensor([0.1307]).view(1, 1, 1).to(tensor.device)
    std = torch.tensor([0.3081]).view(1, 1, 1).to(tensor.device)
    img = tensor * std + mean
    img = img.clamp(0, 1).squeeze(0)
    img = img.permute(1, 2, 0)
    return img.detach().cpu().numpy()


# ==============================================================================
# 2. RDZEŃ OBLICZENIOWY XAI (WSPÓLNY DLA OBRAZÓW I TABEL)
# ==============================================================================
def compute_all_attributions(model, input_tensor, target_class, device="cpu", use_slic=False):
    """Główny silnik obliczający atrybucje dla wszystkich 4 metod z biblioteki Captum."""
    model.eval()
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad = True

    feature_mask = None

    # Obsługa SLIC (tylko dla obrazów)
    if use_slic and len(input_tensor.shape) == 4:
        print("Generowanie maski superpikseli (SLIC)...")
        if input_tensor.shape[1] == 3:  # Imagenette
            img_for_slic = input_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            segments = slic(img_for_slic, n_segments=50, compactness=10, start_label=0)
        else:  # MNIST
            img_for_slic = input_tensor.squeeze().detach().cpu().numpy()
            segments = slic(
                img_for_slic,
                n_segments=150,  # Zwiększone, by LIME mogło lepiej "rzeźbić" kształt   #0.1 #sigma 0.5 150
                compactness=0.1,   #250, 0.01 wylaczone
                sigma=0.5,# BARDZO MAŁA WARTOŚĆ! Pozwala superpikselom trzymać się krawędzi cyfry
                start_label=0,
                channel_axis=None
            )

        feature_mask = torch.tensor(segments).unsqueeze(0).unsqueeze(0).long().to(device)

    print("Obliczanie Saliency...")
    saliency = Saliency(model)
    attr_sal = saliency.attribute(input_tensor, target=target_class)

    print("Obliczanie Integrated Gradients...")
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(input_tensor).to(device)
    attr_ig = ig.attribute(input_tensor, baselines=baseline, target=target_class)

    print("Obliczanie Feature Ablation...")
    ablation = FeatureAblation(model)
    if feature_mask is not None:
        attr_abl = ablation.attribute(input_tensor, target=target_class, feature_mask=feature_mask)
    else:
        attr_abl = ablation.attribute(input_tensor, target=target_class)

    print("Obliczanie LIME...")
    lime = Lime(model, interpretable_model=SkLearnLinearRegression(), similarity_func=None)
    if feature_mask is not None:
        attr_lime = lime.attribute(input_tensor, target=target_class, feature_mask=feature_mask, n_samples=300)
    else:
        attr_lime = lime.attribute(input_tensor, target=target_class, n_samples=150)

    return {
        "Saliency": attr_sal,
        "Integrated_Gradients": attr_ig,
        "Feature_Ablation": attr_abl,
        "LIME": attr_lime
    }


# ==============================================================================
# 3. ANALIZA DANYCH TABELARYCZNYCH (WYKRESY SŁUPKOWE)
# ==============================================================================
def plot_style_bar(attributions, feature_names, title, ylabel, filename):
    """Generuje wykres słupkowy dla atrybucji z automatycznym skalowaniem."""

    width = max(10, len(feature_names) * 0.4)
    fig, ax = plt.subplots(figsize=(width, 6))
    fig.patch.set_facecolor('white')

    bar_color = '#4c72b0'
    ax.bar(
        range(len(attributions)),
        attributions,
        color=bar_color,
        edgecolor='none',
        width=0.6
    )

    ax.set_title(title, fontsize=12, pad=15)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.axhline(0, color='black', linewidth=0.8)

    # Automatyczne skalowanie osi Y
    max_abs = np.max(np.abs(attributions))

    if max_abs > 0:
        margin = max_abs * 0.1
        ax.set_ylim(
            -max_abs - margin,
            max_abs + margin
        )

    ax.grid(axis='y', linestyle='-', alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_edgecolor('#a0a0a0')
        spine.set_linewidth(1.0)

    ax.set_xticks(range(len(feature_names)))

    clean_labels = [
        str(name).split('_')[-1]
        if '_' in str(name)
        else str(name)
        for name in feature_names
    ]

    ax.set_xticklabels(
        clean_labels,
        rotation=45,
        ha='right',
        rotation_mode='anchor',
        fontsize=8
    )

    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(fig)

def analyze_tabular(model, input_tensor, target_class, feature_names, dataset_name="Zbiór_Tabelaryczny", device="cpu",
                    use_slic=False):
    """Analiza na jednym przykładzie dla danych tabelarycznych (zapisywana jako słupki)."""
    output_dir = "scores"
    os.makedirs(output_dir, exist_ok=True)

    if use_slic:
        print("UWAGA: SLIC pomijane dla danych tabelarycznych (nie obrazowych).")

    attributions = compute_all_attributions(model, input_tensor, target_class, device, use_slic=False)

    safe_dataset_name = dataset_name.replace(" ", "_").replace("(", "").replace(")", "")
    if use_slic: safe_dataset_name += "_SLIC_mode"

    for name, attr in attributions.items():
        attr_np = attr.squeeze().detach().cpu().numpy()
        filepath = os.path.join(output_dir, f"{safe_dataset_name}_{name}.png")
        plot_style_bar(attr_np, feature_names, f"{dataset_name} - {name.replace('_', ' ')}", "Atrybucja", filepath)


def generate_comprehensive_tabular_report(model, dataset, feature_names, class_names, dataset_name="Tabular",
                                          device="cpu", include_mean=True):
    """Dzieli dane tabelaryczne (poprawne/błędne) i generuje pełny raport słupkowy dla 4 metod XAI."""
    model.eval()
    print(f"\n[INFO] Generowanie raportów słupkowych dla całego zbioru {dataset_name}...")

    methods = ["Saliency", "Integrated_Gradients", "Feature_Ablation", "LIME"]

    # Inicjalizacja klas atrybucji, by nie tworzyć ich w pętli
    saliency = Saliency(model)
    ig = IntegratedGradients(model)
    ablation = FeatureAblation(model)
    lime = Lime(model, interpretable_model=SkLearnLinearRegression(), similarity_func=None)

    correct_attrs = {m: {i: [] for i in range(len(class_names))} for m in methods}
    incorrect_attrs = {m: {i: [] for i in range(len(class_names))} for m in methods}

    # 1. Zbieranie atrybucji
    for idx in range(len(dataset)):
        features, true_label = dataset[idx]
        features = features.unsqueeze(0).to(device)
        features.requires_grad = True

        output = model(features)
        pred_label = torch.argmax(output, dim=1).item()

        baseline = torch.zeros_like(features).to(device)

        attr_dict = {
            "Saliency": saliency.attribute(features, target=true_label).squeeze().detach().cpu().numpy(),
            "Integrated_Gradients": ig.attribute(features, baselines=baseline,
                                                 target=true_label).squeeze().detach().cpu().numpy(),
            "Feature_Ablation": ablation.attribute(features, target=true_label).squeeze().detach().cpu().numpy(),
            "LIME": lime.attribute(features, target=true_label, n_samples=150).squeeze().detach().cpu().numpy()
        }

        if pred_label == true_label:
            for m in methods: correct_attrs[m][true_label].append(attr_dict[m])
        else:
            for m in methods: incorrect_attrs[m][true_label].append(attr_dict[m])

    # 2. Generowanie wykresów
    for m in methods:
        m_display = m.replace('_', ' ')
        output_dir = os.path.join("scores", f"{dataset_name}_Comprehensive", m)
        print(f" -> Zapisywanie wykresów dla metody: {m_display} w katalogu: {output_dir}")

        for c_idx, c_name in enumerate(class_names):
            safe_c_name = str(c_name).replace(" ", "_")

            if len(correct_attrs[m][c_idx]) > 0:
                plot_style_bar(
                    correct_attrs[m][c_idx][0], feature_names,
                    title=f"{c_name} - Atrybucje - {m_display} (Correct Example)",
                    ylabel=f"Wartość atrybucji",
                    filename=f"{output_dir}/{safe_c_name}_Correct_Single.png"
                )
                if include_mean:
                    plot_style_bar(
                        np.mean(correct_attrs[m][c_idx], axis=0), feature_names,
                        title=f"{dataset_name} - {c_name} - Correct (Mean) - {m_display}",
                        ylabel=f"Średnia atrybucja",
                        filename=f"{output_dir}/{safe_c_name}_Correct_Mean.png"
                    )

            if len(incorrect_attrs[m][c_idx]) > 0:
                plot_style_bar(
                    incorrect_attrs[m][c_idx][0], feature_names,
                    title=f"{c_name} - Atrybucje - {m_display} (Incorrect Example)",
                    ylabel=f"Wartość atrybucji",
                    filename=f"{output_dir}/{safe_c_name}_Incorrect_Single.png"
                )
                if include_mean:
                    plot_style_bar(
                        np.mean(incorrect_attrs[m][c_idx], axis=0), feature_names,
                        title=f"{dataset_name} - {c_name} - Incorrect (Mean) - {m_display}",
                        ylabel=f"Średnia atrybucja",
                        filename=f"{output_dir}/{safe_c_name}_Incorrect_Mean.png"
                    )


# ==============================================================================
# 4. ANALIZA DANYCH OBRAZOWYCH (MAPY CIEPLNE)
# ==============================================================================
def analyze_image(model, input_tensor, target_class, dataset_name="Zbiór_Obrazowy", device="cpu", use_slic=False):
    """Analiza na jednym przykładzie obrazowym z wygenerowaniem kolorowych map cieplnych."""
    output_dir = "scores"
    os.makedirs(output_dir, exist_ok=True)

    attributions = compute_all_attributions(model, input_tensor, target_class, device, use_slic=use_slic)

    safe_dataset_name = dataset_name.replace(" ", "_").replace("(", "").replace(")", "")
    if use_slic: safe_dataset_name += "_SLIC"

    orig_img = unnormalize_imagenette(input_tensor) if input_tensor.shape[1] == 3 else unnormalize_mnist(input_tensor)

    # 1. Zapis obrazu oryginalnego
    dummy_attr = np.zeros_like(orig_img)
    dummy_attr[0, 0, 0] = 1.0  # Ochrona przed błędem scale_factor=0

    color_map_orig = "gray" if orig_img.shape[-1] == 1 else None
    fig_orig, _ = viz.visualize_image_attr(
        attr=dummy_attr, original_image=orig_img, method="original_image",
        cmap=color_map_orig, title="Oryginalny Obraz", use_pyplot=False
    )
    orig_filepath = os.path.join(output_dir, f"{safe_dataset_name}_Original.png")
    fig_orig.savefig(orig_filepath, dpi=300)
    plt.close(fig_orig)
    print(f"Zapisano obraz: {orig_filepath}")

    # 2. Zapis map atrybucji
    for name, attr in attributions.items():
        attr_np = attr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        if np.max(np.abs(attr_np)) == 0: attr_np[0, 0, 0] = 1e-10

        title_suffix = " (SLIC)" if use_slic and name in ["LIME", "Feature_Ablation"] else ""

        fig, _ = viz.visualize_image_attr(
            attr=attr_np, original_image=orig_img, method="heat_map",
            cmap="jet", show_colorbar=True, title=f"{name.replace('_', ' ')}{title_suffix}", use_pyplot=False
        )
        filepath = os.path.join(output_dir, f"{safe_dataset_name}_{name}.png")
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        print(f"Zapisano wykres: {filepath}")


def generate_class_examples_image_report(model, dataset, class_names, dataset_name="Zbiór_Obrazowy", device="cpu",
                                         use_slic=False):
    """Szuka po 1 poprawnym i 1 błędnym przykładzie dla każdej klasy, zapisuje oryginał i generuje mapy."""
    model.eval()
    print(f"\n[INFO] Szukanie przykładów (1 poprawny + 1 błędny) dla klas w zbiorze {dataset_name}...")

    methods = ["Saliency", "Integrated_Gradients", "Feature_Ablation", "LIME"]

    # Inicjalizacja słownika, który przechowa zarówno poprawny, jak i błędny przykład dla każdej klasy
    found_examples = {i: {'correct': None, 'incorrect': None} for i in range(len(class_names))}

    # 1. Przeszukiwanie datasetu w poszukiwaniu pary dla każdej klasy
    with torch.no_grad():  # Znacznie przyspiesza skanowanie
        for idx in range(len(dataset)):
            # Przerywamy, jeśli mamy już komplet (i poprawne i błędne dla wszystkich klas)
            if all(v['correct'] is not None and v['incorrect'] is not None for v in found_examples.values()):
                break

            features, true_label = dataset[idx]

            # Jeśli mamy już oba przykłady dla tej klasy, nie marnujemy czasu na forward
            if found_examples[true_label]['correct'] is not None and found_examples[true_label][
                'incorrect'] is not None:
                continue

            features_batch = features.unsqueeze(0).to(device)
            pred_label = torch.argmax(model(features_batch), dim=1).item()
            is_correct = (pred_label == true_label)

            # Przypisywanie w zależności od tego, czego nam jeszcze brakuje
            if is_correct and found_examples[true_label]['correct'] is None:
                orig_img = unnormalize_imagenette(features_batch) if features_batch.shape[
                                                                         1] == 3 else unnormalize_mnist(features_batch)
                found_examples[true_label]['correct'] = (features_batch, orig_img, pred_label)
                print(f"  Znaleziono: {class_names[true_label]:<12} [Poprawna] (Próbka: {idx})")

            elif not is_correct and found_examples[true_label]['incorrect'] is None:
                orig_img = unnormalize_imagenette(features_batch) if features_batch.shape[
                                                                         1] == 3 else unnormalize_mnist(features_batch)
                found_examples[true_label]['incorrect'] = (features_batch, orig_img, pred_label)
                print(
                    f"  Znaleziono: {class_names[true_label]:<12} [Błędna jako {class_names[pred_label]}] (Próbka: {idx})")

    safe_dataset_name = dataset_name.replace(" ", "_").replace("(", "").replace(")", "")
    if use_slic: safe_dataset_name += "_SLIC"
    slic_suffix = "_SLIC" if use_slic else ""

    print("\n[INFO] Obliczanie atrybucji i generowanie wykresów...")

    # 2. Generowanie wykresów dla znalezionych przykładów
    for c_idx, examples in found_examples.items():
        c_name = class_names[c_idx]
        safe_c_name = str(c_name).replace(" ", "_")

        # Dla każdej klasy analizujemy zarówno klucz 'correct', jak i 'incorrect'
        for status_key, data in examples.items():
            if data is None:
                continue  # Model ma np. 100% skuteczności i nie wygenerował błędów dla tej klasy

            features_batch, orig_img, pred_label = data

            is_correct = (status_key == 'correct')
            # Tworzymy ładne etykiety do plików i tytułów
            status_file = "Correct" if is_correct else f"Incorrect_as_{class_names[pred_label]}"
            status_title = "Correct" if is_correct else f"Incorrect (Pred: {class_names[pred_label]})"

            print(f" -> Przetwarzanie klasy: {c_name} [{status_title}] ...")

            # --- Zapis oryginału ---
            output_dir_orig = os.path.join("scores", f"{safe_dataset_name}_Classes", "Original")
            os.makedirs(output_dir_orig, exist_ok=True)

            dummy_attr = np.zeros_like(orig_img)
            dummy_attr[0, 0, 0] = 1.0
            color_map_orig = "gray" if orig_img.shape[-1] == 1 else None

            fig_orig, _ = viz.visualize_image_attr(
                attr=dummy_attr, original_image=orig_img, method="original_image",
                cmap=color_map_orig, title=f"Oryginał - {c_name} [{status_title}]", use_pyplot=False
            )
            filepath_orig = os.path.join(output_dir_orig, f"{safe_c_name}_{status_file}_Original.png")
            fig_orig.savefig(filepath_orig, dpi=300)
            plt.close(fig_orig)

            # --- Obliczanie i zapis atrybucji dla 4 metod ---
            attributions = compute_all_attributions(model, features_batch, c_idx, device=device, use_slic=use_slic)

            for m in methods:
                output_dir = os.path.join("scores", f"{safe_dataset_name}_Classes", m)
                os.makedirs(output_dir, exist_ok=True)

                attr_np = attributions[m].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                if np.max(np.abs(attr_np)) == 0: attr_np[0, 0, 0] = 1e-10

                # --- ZMIANA: Tytuł zależy od tego, czy metoda obsługuje SLIC ---
                if use_slic and m in ["LIME", "Feature_Ablation"]:
                    title_text = f"{c_name} - {m.replace('_', ' ')}{slic_suffix}\n[{status_title}]"
                else:
                    title_text = f"{c_name} - {m.replace('_', ' ')}\n[{status_title}]"

                fig, _ = viz.visualize_image_attr(
                    attr=attr_np, original_image=orig_img, method="heat_map",
                    sign="all", show_colorbar=True, use_pyplot=False,
                    title=title_text  # Używamy przygotowanego tytułu
                )

                filepath = os.path.join(output_dir, f"{safe_c_name}_{status_file}_{m}{slic_suffix}.png")
                fig.savefig(filepath, dpi=300)
                plt.close(fig)


# ==============================================================================
# 5. WYJAŚNIENIA KONTRFAKTYCZNE (PERTURBATION / ZASŁANIANIE PIKSELI)
# ==============================================================================
# ==============================================================================
# 5. WYJAŚNIENIA KONTRFAKTYCZNE (PERTURBATION / ZASŁANIANIE PIKSELI)
# ==============================================================================
def mask_bottom_pixels(image_tensor, lines_to_mask):
    """Funkcja atakująca: zasłania N dolnych wierszy obrazka wartością tła."""
    perturbed_image = image_tensor.clone()
    lines = int(lines_to_mask.item()) if torch.is_tensor(lines_to_mask) else int(lines_to_mask)

    if lines > 0:
        background_value = perturbed_image.min()
        perturbed_image[..., -lines:, :] = background_value

    return perturbed_image


def analyze_and_plot_perturbation(model, image_tensor, original_class, dataset_name="MNIST", device="cpu",
                                  model_name="CNN"):
    """Tworzy wykres ścieżki decyzyjnej i używa Captum MinParamPerturbation do znalezienia załamania."""
    model.eval()
    image_tensor = image_tensor.to(device)

    # Część A: Zbieranie danych do wykresu liniowego
    mask_values = list(range(29))
    predictions = []

    with torch.no_grad():
        for lines in mask_values:
            perturbed_img = mask_bottom_pixels(image_tensor, lines)
            pred = model(perturbed_img.unsqueeze(0)).argmax(dim=1).item()
            predictions.append(pred)

    # Część B: Rysowanie wykresu liniowego
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mask_values, predictions, color='#4c72b0', linestyle='-')

    # --- ZMIANA 1: Nowy tytuł na wykresie ---
    ax.set_title(f"{model_name} - {dataset_name} - Zasłanianie od dołu a predykcja (Klasa {original_class})",
                 fontsize=11)
    ax.set_xlabel("Liczba zasłoniętych wierszy od dołu", fontsize=10)
    ax.set_ylabel("Predykowana klasa", fontsize=10)
    ax.set_xticks(range(0, 30, 5))

    plt.tight_layout()
    output_dir = "scores"
    os.makedirs(output_dir, exist_ok=True)

    # --- ZMIANA 2: Dodanie nazwy modelu (CNN) do zapisywanego pliku ---
    filename = os.path.join(output_dir, f"{dataset_name}_Perturbation_Class_{original_class}_{model_name}.png")
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"\n[INFO] Zapisano wykres perturbacji: {filename}")

    # Część C: Oficjalne poszukiwanie punktu granicznego w Captum
    perturbator = MinParamPerturbation(
        forward_func=model,
        attack=mask_bottom_pixels,
        arg_name="lines_to_mask",
        mode="linear",
        arg_min=0,
        arg_max=28,
        arg_step=1
    )

    try:
        perturbed_image, min_lines = perturbator.evaluate(
            inputs=image_tensor.unsqueeze(0),
            target=original_class,
            perturbations_per_eval=1
        )
        print(f" -> Model zmienił zdanie po zasłonięciu {min_lines} wierszy od dołu.")
    except Exception as e:
        print(f" -> Nie udało się znaleźć granicy (model nie zmienił zdania) lub wystąpił błąd: {e}")


def generate_all_perturbation_reports(model, dataset, class_names, dataset_name="MNIST", device="cpu",
                                      model_name="CNN"):
    """Automatycznie przeszukuje dataset, znajduje po 1 cyfrze z każdej klasy i rysuje wykresy perturbacji."""
    model.eval()
    print(f"\n[INFO] Generowanie wykresów perturbacji dla wszystkich klas w zbiorze {dataset_name}...")

    found_classes = set()

    with torch.no_grad():
        for idx in range(len(dataset)):
            # Jeśli znaleźliśmy już po jednym przykładzie dla wszystkich klas - kończymy
            if len(found_classes) == len(class_names):
                break

            features, true_label = dataset[idx]

            # Jeśli już mamy tę klasę, idziemy dalej
            if true_label in found_classes:
                continue

            features_batch = features.unsqueeze(0).to(device)
            pred_label = model(features_batch).argmax(dim=1).item()

            # Bierzemy tylko przykłady, na których model początkowo się nie pomylił
            if pred_label == true_label:
                found_classes.add(true_label)
                # Uruchamiamy rysowanie wykresu dla tego konkretnego obrazka
                analyze_and_plot_perturbation(
                    model=model,
                    image_tensor=features,
                    original_class=true_label,
                    dataset_name=dataset_name,
                    device=device,
                    model_name=model_name
                )
    print("\n[INFO] Zakończono generowanie wszystkich wykresów perturbacji.")