// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Internal.Utilities
{
   internal static partial class Utils
   {
      public static void Swap<T>(ref T a, ref T b)
      {
         T temp = a;
         a = b;
         b = temp;
      }

      public static int Size(StringBuilder x)
      {
         Contracts.AssertValueOrNull(x);
         return x == null ? 0 : x.Length;
      }
      public static int Size(Array x)
      {
         Contracts.AssertValueOrNull(x);
         return x == null ? 0 : x.Length;
      }

      public static void Shuffle<T>(Random rand, Span<T> rgv)
      {
         Contracts.AssertValue(rand);

         for (int iv = 0; iv < rgv.Length; iv++)
            Swap(ref rgv[iv], ref rgv[iv + rand.Next(rgv.Length - iv)]);
      }

      /// <summary>
      /// A one-argument version of <see cref="MarshalInvoke{TTarget, TResult}(FuncInstanceMethodInfo1{TTarget, TResult}, TTarget, Type)"/>.
      /// </summary>
      public static TResult MarshalInvoke<TTarget, TArg1, TResult>(FuncInstanceMethodInfo1<TTarget, TArg1, TResult> func, TTarget target, Type genArg, TArg1 arg1)
          where TTarget : class
      {
         var meth = func.MakeGenericMethod(genArg);
         return (TResult)meth.Invoke(target, new object[] { arg1 });
      }
   }
}
