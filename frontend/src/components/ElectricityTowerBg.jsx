import React from 'react';

export default function ElectricityTowerBg() {
  return (
    <div 
      aria-hidden="true" 
      className="fixed left-0 top-16 bottom-0 w-[280px] sm:w-[360px] lg:w-[440px] xl:w-[500px] pointer-events-none z-0 overflow-hidden select-none transition-opacity duration-700"
    >
      {/* 3D Volumetric Tower with Alpha Transparent Background */}
      <div className="relative w-full h-full flex items-end justify-start">
        {/* Light theme: Dark blue tower */}
        <img
          src="/tower-light.png"
          alt="3D High-Voltage Transmission Tower (Light Theme)"
          className="w-full h-[90%] max-h-[920px] object-contain object-left-bottom 
                     opacity-40 dark:hidden
                     filter brightness-100 contrast-110
                     transition-opacity duration-500"
        />
        {/* Dark theme: Electric cyan/blue tower */}
        <img
          src="/tower-dark.png"
          alt="3D High-Voltage Transmission Tower (Dark Theme)"
          className="w-full h-[90%] max-h-[920px] object-contain object-left-bottom 
                     opacity-45 hidden dark:block
                     filter brightness-105 contrast-110
                     transition-opacity duration-500"
        />
      </div>
    </div>
  );
}
