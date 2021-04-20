// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.CompilerServices;

namespace Microsoft.ML.Internal.Utilities
{
   internal static partial class Utils
   {
      private const int CbitUint = 32;

      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      public static uint GetLo(ulong uu)
      {
         return (uint)uu;
      }

      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      public static uint GetHi(ulong uu)
      {
         return (uint)(uu >> CbitUint);
      }
   }
}
