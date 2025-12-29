<script lang="ts">
  import { onMount } from 'svelte';
  import { referenceImageStore, referenceImageSent } from '$lib/mediaStream';

  export let width = 512;
  export let height = 512;

  let fileInputEl: HTMLInputElement;
  let canvasEl: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  let previewUrl: string | null = null;
  let imgEl: HTMLImageElement;

  onMount(() => {
    ctx = canvasEl.getContext('2d')!;
    canvasEl.width = width;
    canvasEl.height = height;
  });

  export function clear() {
    // 1. 释放内存并清空预览 URL
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }
    
    // 2. 清空 Canvas 画面
    if (ctx) {
        ctx.clearRect(0, 0, width, height);
    }

    // 3. 重置文件输入框
    if (fileInputEl) {
        fileInputEl.value = '';
    }

    // 4. 传入一个空的 Blob 对象，而不是 null
    referenceImageStore.set({ blob: new Blob([]) });
    
    referenceImageSent.set(false);
  }

  export function loadBlob(blob: Blob) {
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl); // 释放旧内存
    }
    previewUrl = URL.createObjectURL(blob);
    // 注意：赋值给 previewUrl 后，Svelte 会渲染 <img> 标签
    // <img> 加载完成后会自动触发下方的 on:load={process}，无需手动调用 process()
  }

  function onSelect(e: Event) {
    const input = e.target as HTMLInputElement;
    const f = input.files?.[0];
    if (!f) return;

    loadBlob(f);
  }

  async function process() {
    const w = imgEl.naturalWidth;
    const h = imgEl.naturalHeight;
    const side = Math.min(w, h);
    const x = (w - side) / 2;
    const y = (h - side) / 2;

    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(imgEl, x, y, side, side, 0, 0, width, height);

    const blob = await new Promise<Blob>((resolve) =>
      canvasEl.toBlob((b) => resolve(b!), "image/jpeg", 1)
    );

    referenceImageStore.set({ blob });

    referenceImageSent.set(false);

    // previewUrl = URL.createObjectURL(blob);
  }
</script>

<!-- 全部内容与 VideoInput 外层框架统一 -->
<div class="relative mx-auto aspect-square w-full overflow-hidden rounded-lg border border-slate-300">

  <!-- 文件选择控件：悬停在左上角，不占布局高度 -->
  <div class="absolute bottom-2 left-2 z-20 bg-black/50 text-white px-1 py-0.5 font-medium rounded-md text-[10px] backdrop-blur-sm shadow">

    <!-- 自定义的小按钮 -->
    <label
      for="refImageUpload"
      class="cursor-pointer select-none"
    >
      ① Upload
    </label>

    <!-- 隐藏原生 input -->
    <input
      id="refImageUpload"
      type="file"
      accept="image/*"
      class="hidden"
      bind:this={fileInputEl}
      on:change={onSelect}
    />
  </div>
  <div
    class="absolute top-2 left-2 z-30 bg-black/50 text-white 
            text-xs font-medium px-2 py-0.5 rounded-md backdrop-blur-sm shadow"
  >
    🧙 Portrait
  </div>

  <!-- 正方形预览（裁剪后） -->
  {#if previewUrl}
    <img
      bind:this={imgEl}
      src={previewUrl}
      alt="preview"
      class="w-full h-full object-cover"
      on:load={process}
    />
  {:else}
    <div class="flex items-center justify-center w-full h-full bg-transparent text-gray-400 dark:text-gray-500">
      <svg xmlns="http://www.w3.org/2000/svg" class="w-28 opacity-30" viewBox="0 0 448 512">
        <path fill="currentColor"
          d="M224 256A128 128 0 1 0 224 0a128 128 0 1 0  0 256zm-45.7 48A178.3 178.3 0 0 0 0 482.3 29.7 29.7 0 0 0 29.7 512h388.6A29.7 29.7 0 0 0 448 482.3c0-98.5-79.8-178.3-178.3-178.3z"/>
      </svg>
    </div>
  {/if}

  <canvas bind:this={canvasEl} class="hidden"></canvas>
</div>